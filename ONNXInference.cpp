// ONNXInference.cpp
#include "ONNXInference.h"

#include <stdexcept>
#include <codecvt>
#include <chrono>
#include <iostream>
#include <thread>
#include <algorithm>

namespace {

// Cross-platform model path helper for ORT ctor
#ifdef _WIN32
inline std::wstring to_wpath(const std::string& s){
  return std::wstring_convert<std::codecvt_utf8<wchar_t>>{}.from_bytes(s);
}
#endif

} // namespace

// ===== ctor =============================================================
ONNXInference::ONNXInference(const std::string& model_path, const Options& opts)
  : model_path_(model_path),
    opts_(opts),
    env_(ORT_LOGGING_LEVEL_WARNING, "ONNXInference"),
    sessionOptions_(),
    session_(nullptr),
    allocator_(),
    memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

  applySessionOptions();
  loadModel(model_path_);
  inspectIO();
}

// ===== public: inference ===============================================
std::vector<Tensor> ONNXInference::infer(const std::vector<std::vector<float>>& inTensors) {
  if (!session_) throw std::runtime_error("infer(): session not initialized");
  if (inTensors.size() != input_names_.size())
    throw std::runtime_error("infer(): input count mismatch");

  std::vector<Ort::Value> ortInputs;
  ortInputs.reserve(inTensors.size());

  for (size_t i = 0; i < inTensors.size(); ++i) {
    const auto &st = input_shapes_[i];

    // Compute product of known dims, count dynamics (<=0)
    size_t prodKnown = 1; int dyn = 0;
    for (auto d : st) {
      if (d > 0) prodKnown *= static_cast<size_t>(d);
      else       ++dyn;
    }

    const size_t blobSz = inTensors[i].size();

    std::vector<int64_t> runShape = st;
    if (dyn > 0) {
      if (prodKnown == 0 || (blobSz % prodKnown) != 0)
        throw std::runtime_error("infer(): cannot infer dynamic dimension — blob size not divisible by known product");
      size_t fill = blobSz / prodKnown;

      bool filled = false;
      for (auto &d : runShape) {
        if (d <= 0) { d = static_cast<int64_t>(fill); filled = true; break; }
      }
      if (!filled)
        throw std::runtime_error("infer(): internal — dynamic dim not found to fill");
    } else {
      if (blobSz != prodKnown)
        throw std::runtime_error("infer(): tensor size mismatch for fixed-shape input");
    }

    ortInputs.push_back(
      Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(inTensors[i].data()),
        blobSz,
        runShape.data(), runShape.size()
      )
    );
  }

  std::vector<const char*> inNames, outNames;
  inNames.reserve(input_names_.size());
  outNames.reserve(output_names_.size());
  for (auto &n : input_names_)  inNames.push_back(n.c_str());
  for (auto &n : output_names_) outNames.push_back(n.c_str());

  const auto t0 = std::chrono::high_resolution_clock::now();

  auto ortOut = session_->Run(
    Ort::RunOptions{nullptr},
    inNames.data(),  ortInputs.data(), static_cast<size_t>(ortInputs.size()),
    outNames.data(), static_cast<size_t>(outNames.size())
  );

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[TIMING] ORT Run took " << elapsed_ms << " ms  (EP=" << activeProvider() << ")\n";

  std::vector<Tensor> results;
  results.reserve(ortOut.size());
  for (auto &o : ortOut) {
    Tensor t;
    auto info = o.GetTensorTypeAndShapeInfo();
    t.shape = info.GetShape();
    size_t cnt = info.GetElementCount();
    float* p = o.GetTensorMutableData<float>();
    t.data.assign(p, p + cnt);
    results.push_back(std::move(t));
  }
  return results;
}

// ===== public: shape helpers ===========================================
void ONNXInference::fixDynamicHW(int newH, int newW) {
  for (auto &shp : input_shapes_) {
    if (shp.size() >= 4) {  // assume N,C,H,W
      shp[2] = newH;
      shp[3] = newW;
    }
  }
}

void ONNXInference::fixExactInputShape(size_t index, const std::vector<int64_t>& shape) {
  if (index >= input_shapes_.size()) throw std::runtime_error("fixExactInputShape(): bad input index");
  input_shapes_[index] = shape;
}

// ===== public: runtime tuning ==========================================
void ONNXInference::setThreads(int n){ setIntraOpThreads(n); }

void ONNXInference::setIntraOpThreads(int n){
  n = std::max(1, n);
  if (opts_.intra_threads == n) return;
  opts_.intra_threads = n;
  markDirtyAndMaybeRebuild(true);
}

void ONNXInference::setInterOpThreads(int n){
  n = std::max(1, n);
  if (opts_.inter_threads == n) return;
  opts_.inter_threads = n;
  markDirtyAndMaybeRebuild(true);
}

void ONNXInference::setGraphOptimization(GraphOptimizationLevel lvl){
  if (opts_.graph_opt == lvl) return;
  opts_.graph_opt = lvl;
  markDirtyAndMaybeRebuild(true);
}

void ONNXInference::enableArena(bool on){
  if (opts_.enable_arena == on) return;
  opts_.enable_arena = on;
  markDirtyAndMaybeRebuild(true);
}

void ONNXInference::enableMemPattern(bool on){
  if (opts_.enable_mem_pattern == on) return;
  opts_.enable_mem_pattern = on;
  markDirtyAndMaybeRebuild(true);
}

void ONNXInference::useOpenVINO(bool on, const std::string& device){
  if (opts_.use_openvino_ep == on && (!on || opts_.openvino_device == device)) return;
  opts_.use_openvino_ep = on;
  if (on) opts_.openvino_device = device;
  markDirtyAndMaybeRebuild(true);
}

void ONNXInference::warmup(const std::vector<std::vector<float>>& inputs){
  // Best-effort: just call infer and discard outputs.
  try { (void)infer(inputs); } catch (...) { /* ignore warmup errors */ }
}

void ONNXInference::rebuild(){
  applySessionOptions();
  loadModel(model_path_);
  inspectIO();
  dirty_ = false;
}

// ===== private: core helpers ===========================================
void ONNXInference::markDirtyAndMaybeRebuild(bool force){
  dirty_ = true;
  if (force) rebuild();
}

void ONNXInference::applySessionOptions(){
  // Recreate options object each time to clear previous EPs/flags.
  sessionOptions_ = Ort::SessionOptions();

  // Graph optimizations
  sessionOptions_.SetGraphOptimizationLevel(opts_.graph_opt);

  // Latency-friendly defaults
  sessionOptions_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  sessionOptions_.SetIntraOpNumThreads(std::max(1, opts_.intra_threads));
  sessionOptions_.SetInterOpNumThreads(std::max(1, opts_.inter_threads));

  // Memory controls
  if (!opts_.enable_arena)      sessionOptions_.DisableCpuMemArena();
  if (!opts_.enable_mem_pattern) sessionOptions_.DisableMemPattern();

  active_provider_ = "CPUExecutionProvider";
#if defined(USE_ORT_OPENVINO_EP)
  if (opts_.use_openvino_ep) {
    // Attach OpenVINO EP with the requested device (e.g., "CPU", "AUTO:CPU")
    OrtSessionOptionsAppendExecutionProvider_OpenVINO(sessionOptions_, opts_.openvino_device.c_str());
    active_provider_ = std::string("OpenVINOExecutionProvider(") + opts_.openvino_device + ")";
  }
#endif
}

void ONNXInference::loadModel(const std::string& model_path) {
#ifdef _WIN32
  std::wstring wpath = to_wpath(model_path);
  session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), sessionOptions_);
#else
  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), sessionOptions_);
#endif
}

void ONNXInference::inspectIO() {
  if (!session_) throw std::runtime_error("inspectIO(): session null");

  size_t nIn = session_->GetInputCount();
  input_names_.resize(nIn);
  input_shapes_.resize(nIn);
  for (size_t i = 0; i < nIn; ++i) {
    auto np = session_->GetInputNameAllocated(i, allocator_);
    input_names_[i] = np.get();
    auto info = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    input_shapes_[i] = info.GetShape();
  }

  size_t nOut = session_->GetOutputCount();
  output_names_.resize(nOut);
  output_shapes_.resize(nOut);
  for (size_t i = 0; i < nOut; ++i) {
    auto np = session_->GetOutputNameAllocated(i, allocator_);
    output_names_[i] = np.get();
    auto info = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
    output_shapes_[i] = info.GetShape();
  }
}


// Keep: patch dynamic H/W after computing in main.cpp (for single-input cases)
void ONNXInference::fixDynamicHW(int newH, int newW) {
  for (auto &shp : input_shapes_) {
    if (shp.size() >= 4) {  // assume N,C,H,W
      shp[2] = newH;
      shp[3] = newW;
    }
  }
}

// NEW: allow caller to force an exact input shape (e.g., for batched OCR: [B,3,H,Wmax])
void ONNXInference::fixExactInputShape(size_t index, const std::vector<int64_t>& shape) {
  if (index >= input_shapes_.size()) throw std::runtime_error("fixExactInputShape: bad input index");
  input_shapes_[index] = shape;
}

std::vector<Tensor> ONNXInference::infer(const std::vector<std::vector<float>>& inTensors) {
  if (inTensors.size() != input_names_.size())
    throw std::runtime_error("Input count mismatch");

  std::vector<Ort::Value> ortInputs;
  ortInputs.reserve(inTensors.size());

  for (size_t i = 0; i < inTensors.size(); ++i) {
    const auto &st = input_shapes_[i];

    // Compute product of known dims, count dynamics (<=0)
    size_t prodKnown = 1; int dyn = 0;
    for (auto d : st) {
      if (d > 0) prodKnown *= static_cast<size_t>(d);
      else       ++dyn;
    }

    const size_t blobSz = inTensors[i].size();

    // If no dynamic dims, enforce exact size
    if (dyn == 0) {
      if (blobSz != prodKnown)
        throw std::runtime_error("Tensor size mismatch for fixed-shape input");
      ortInputs.push_back(
        Ort::Value::CreateTensor<float>(
          memory_info_,
          const_cast<float*>(inTensors[i].data()),
          blobSz,
          st.data(), st.size()
        )
      );
      continue;
    }

    // If we have dynamics: fill the FIRST dynamic dim with the implied size.
    // NOTE: for multiple dynamics you should call fixExactInputShape() before infer().
    auto runShape = st;
    if (dyn >= 1) {
      if (prodKnown == 0 || (blobSz % prodKnown) != 0)
        throw std::runtime_error("Cannot infer dynamic dimension: blob size not divisible by known product");
      size_t fill = blobSz / prodKnown;

      bool filled = false;
      for (auto &d : runShape) {
        if (d <= 0) { d = static_cast<int64_t>(fill); filled = true; break; }
      }
      if (!filled)
        throw std::runtime_error("Internal error: dynamic dim not found to fill");
    }

    ortInputs.push_back(
      Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(inTensors[i].data()),
        blobSz,
        runShape.data(), runShape.size()
      )
    );
  }

  std::vector<const char*> inNames, outNames;
  inNames.reserve(input_names_.size());
  outNames.reserve(output_names_.size());
  for (auto &n : input_names_)  inNames.push_back(n.c_str());
  for (auto &n : output_names_) outNames.push_back(n.c_str());

  // Timing around the Run
  auto t0 = std::chrono::high_resolution_clock::now();

  auto ortOut = session_->Run(
    Ort::RunOptions{nullptr},
    inNames.data(),  ortInputs.data(), static_cast<size_t>(ortInputs.size()),
    outNames.data(), static_cast<size_t>(outNames.size())
  );

  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "[TIMING] ORT Session::Run took " << elapsed_ms << " ms\n";

  std::vector<Tensor> results;
  results.reserve(ortOut.size());
  for (auto &o : ortOut) {
    Tensor t;
    auto info = o.GetTensorTypeAndShapeInfo();
    t.shape = info.GetShape();
    size_t cnt = info.GetElementCount();
    float* p = o.GetTensorMutableData<float>();
    t.data.assign(p, p + cnt);
    results.push_back(std::move(t));
  }
  return results;
}
