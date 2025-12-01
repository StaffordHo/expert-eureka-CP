#pragma once
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// Force the API version to match your installed ORT (e.g., 1.17.x)
#ifndef ORT_API_VERSION
#  define ORT_API_VERSION 17
#endif

#include <onnxruntime_cxx_api.h>

// === Optional: OpenVINO Execution Provider for ONNX Runtime ============
// Define USE_ORT_OPENVINO_EP in your build if you link the OpenVINO EP.
//   MSVC: /DUSE_ORT_OPENVINO_EP
//   CMake: add_definitions(-DUSE_ORT_OPENVINO_EP)
// Include path may differ depending on your ORT package layout.
#if defined(USE_ORT_OPENVINO_EP)
#  include "onnxruntime_openvino_provider_factory.h"
#endif
// ======================================================================

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float>   data;
};

class ONNXInference {
public:
  // High-level options bag (sane CPU defaults for latency-first)
  struct Options {
    int  intra_threads = 1;
    int  inter_threads = 1;
    GraphOptimizationLevel graph_opt = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
    bool enable_arena      = true;
    bool enable_mem_pattern= true;

    // Execution Provider (EP) selection
    bool        use_openvino_ep = false;   // if false => default CPU EP
    std::string openvino_device = "CPU";   // "CPU", "AUTO:CPU", "GPU" (if present)
  };

  explicit ONNXInference(const std::string& model_path,
                         const Options& opts = Options{});

  // One-shot inference. Shapes are taken from fixExactInputShape()/fixDynamicHW() or initial IO.
  std::vector<Tensor> infer(const std::vector<std::vector<float>>& input_tensors);

  // ---- Shape management ------------------------------------------------
  // Patch the first (HxW) of the first input AFTER you compute them (convenience for typical NCHW).
  void fixDynamicHW(int newH, int newW);

  // Force an exact input shape for a given input index (e.g., rec bucketing): {1,3,H,W}
  void fixExactInputShape(size_t index, const std::vector<int64_t>& shape);

  // ---- Runtime tuning (all trigger a session rebuild when changed) ----
  // Intra-op threads (compute parallelism). Keeps inter-op the same.
  void setThreads(int n); // backward-compatible alias for setIntraOpThreads

  void setIntraOpThreads(int n);
  void setInterOpThreads(int n);
  void setGraphOptimization(GraphOptimizationLevel lvl);
  void enableArena(bool on);
  void enableMemPattern(bool on);

  // Switch EPs at runtime (requires rebuild)
  // Example: useOpenVINO(true, "CPU"); or useOpenVINO(false) to go back to CPU EP.
  void useOpenVINO(bool on, const std::string& device = "CPU");

  // Force rebuilding the session with current options/shapes/EP.
  void rebuild();

  // Optional: run one dummy pass to trigger JIT/fusions, caches, and EP compilation.
  // Provide one tensor per input; pass small dummy buffers with correct shapes.
  void warmup(const std::vector<std::vector<float>>& input_tensors);

  // ---- Introspection ---------------------------------------------------
  const std::vector<std::string>&            inputNames()   const { return input_names_; }
  const std::vector<std::vector<int64_t>>&   inputShapes()  const { return input_shapes_; }
  const std::vector<std::string>&            outputNames()  const { return output_names_; }
  const std::vector<std::vector<int64_t>>&   outputShapes() const { return output_shapes_; }

  // Convenience: which EP is active
  std::string activeProvider() const { return active_provider_; }

private:
  // Core helpers
  void loadModel(const std::string& model_path);  // (re)creates session_ with sessionOptions_
  void inspectIO();                               // fills input/output names & shapes
  void applySessionOptions();                     // applies Options -> sessionOptions_
  void markDirtyAndMaybeRebuild(bool force=false);

  // Model & configuration
  std::string   model_path_;
  Options       opts_;
  bool          dirty_ = false;       // need to rebuild session

  // ORT core objects
  Ort::Env                          env_;
  Ort::SessionOptions               sessionOptions_;
  std::unique_ptr<Ort::Session>     session_;
  Ort::AllocatorWithDefaultOptions  allocator_;
  Ort::MemoryInfo                   memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  // I/O metadata (kept in sync after rebuild/reshape)
  std::vector<std::string>             input_names_;
  std::vector<std::vector<int64_t>>    input_shapes_;
  std::vector<std::string>             output_names_;
  std::vector<std::vector<int64_t>>    output_shapes_;

  // Current EP label (for introspection / logs)
  std::string active_provider_ = "CPUExecutionProvider";
};
