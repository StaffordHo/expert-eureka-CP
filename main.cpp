// main.cpp
#include "ONNXInference.h"

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>  // NMSBoxes

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>      // std::memcpy
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX       // prevent windows.h min/max macros
#endif
#include <windows.h>
#endif

// ======= knobs you can tweak =======
static constexpr bool  USE_DB_ROTATED_BOX   = true;    // rotated box + unclip
static constexpr bool  USE_PERSPECTIVE_CROP = true;    // perspective rectify crops
static constexpr float DB_BIN_THRESH        = 0.38f;   // 0.35–0.45 typical for dot-peen
static constexpr int   DB_MIN_AREA          = 1500;    // filter tiny contours
static constexpr float DB_UNCLIP_RATIO      = 1.5f;    // expand polygons a bit
static constexpr int   REC_MAX_W            = 512;     // allow wider lines
static constexpr bool  REC_USE_PPOCR_NORM   = true;    // mean=std=0.5 on [0,1]
static constexpr float CTC_REJECT_AVGCONF   = 0.30f;   // reject reads below this
static constexpr bool  APPLY_DOTPEEN_ENH    = true;    // CLAHE + tophat + bin
// ====================================

// ======== args / csv / metrics =========================================
struct Args {
    std::string detPath, recPath, keysPath, singleImage, listPath, csvPath = "bench_results.csv", labelsPath;
    int runs = 30, warmup = 5, threads = 1;
    bool runs_set = false, warmup_set = false;   // NEW
};

static inline bool file_exists(const std::string& p){ std::ifstream f(p); return f.good(); }

static Args parse_args(int argc, char** argv){
    Args a;
    for (int i=1;i<argc;i++){
        std::string s = argv[i];
        auto next = [&](const char* flag){ if (i+1>=argc) { std::cerr<<"Missing value for "<<flag<<"\n"; exit(2);} return std::string(argv[++i]); };
        if      (s == "--images")  a.listPath  = next("--images");
        else if (s == "--runs")    { a.runs      = std::stoi(next("--runs"));   a.runs_set   = true; }
        else if (s == "--warmup")  { a.warmup    = std::stoi(next("--warmup")); a.warmup_set = true; }
        else if (s == "--threads") a.threads   = std::stoi(next("--threads"));
        else if (s == "--csv")     a.csvPath   = next("--csv");
        else if (s == "--labels")  a.labelsPath= next("--labels");
        else if (s == "--keys")    a.keysPath  = next("--keys");
        else if (a.detPath.empty()) a.detPath = s;
        else if (a.recPath.empty() && file_exists(s) && s.size()>=5 && s.substr(s.size()-5)==".onnx") a.recPath = s;
        else a.singleImage = s;
    }
    return a;
}

struct CSV {
    std::ofstream f;
    explicit CSV(const std::string& path){
        bool newfile = !file_exists(path);
        f.open(path, std::ios::app);
        if (!f) { std::cerr<<"[ERROR] cannot open csv "<<path<<"\n"; exit(2); }
        if (newfile){
            f<<"ts_us,image,run_idx,warm,threads,"
             <<"det_pre_ms,det_run_ms,det_decode_ms,"
             <<"rec_total_ms,num_boxes,total_ms,text_concat,"
             <<"cer,wer\n";
        }
    }
    static std::string esc(const std::string& s){
        std::string t; t.reserve(s.size()+8);
        t.push_back('"');
        for(char c: s){ if (c=='"') t.push_back('"'); t.push_back(c); }
        t.push_back('"');
        return t;
    }
    void line(uint64_t ts, const std::string& img, int r, int warm, int threads,
              double det_pre, double det_run, double det_dec,
              double rec_tot, int n_boxes, double total,
              const std::string& text_concat,
              double cer, double wer){
        f<<ts<<","<<esc(img)<<","<<r<<","<<warm<<","<<threads<<","
         <<det_pre<<","<<det_run<<","<<det_dec<<","
         <<rec_tot<<","<<n_boxes<<","<<total<<","<<esc(text_concat)<<","
         <<cer<<","<<wer<<"\n";
        f.flush();
    }
};

static inline uint64_t now_us(){
    using clk=std::chrono::high_resolution_clock;
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(clk::now().time_since_epoch()).count();
}

// Levenshtein distance
static int lev(const std::string& a, const std::string& b){
    const int n=(int)a.size(), m=(int)b.size();
    std::vector<int> dp(m+1); for(int j=0;j<=m;++j) dp[j]=j;
    for(int i=1;i<=n;++i){
        int prev=dp[0]; dp[0]=i;
        for(int j=1;j<=m;++j){
            int tmp=dp[j];
            int cost = (a[i-1]==b[j-1])?0:1;
            dp[j] = std::min({ dp[j]+1, dp[j-1]+1, prev+cost });
            prev=tmp;
        }
    }
    return dp[m];
}

struct GroundTruth {
    std::unordered_map<std::string,std::string> map;
    explicit GroundTruth(const std::string& csv){
        if (csv.empty()) return;
        std::ifstream in(csv);
        if (!in) { std::cerr<<"[WARN] cannot open labels "<<csv<<"\n"; return; }
        std::string line;
        while(std::getline(in,line)){
            if(line.empty()) continue;
            auto pos = line.find(',');
            if(pos==std::string::npos) continue;
            std::string k = line.substr(0,pos);
            std::string v = line.substr(pos+1);
            map[k]=v;
        }
    }
    std::string get(const std::string& img) const {
        auto it=map.find(img); return (it==map.end())?std::string():it->second;
    }
};

static std::vector<std::string> split_words(const std::string& s){
    std::vector<std::string> v; std::string cur;
    for(char c: s){
        if (std::isspace((unsigned char)c)){ if(!cur.empty()){ v.push_back(cur); cur.clear(); } }
        else cur.push_back(c);
    }
    if(!cur.empty()) v.push_back(cur);
    return v;
}

// ───── timing helper ───────────────────────────────────────────────────
struct ScopedTimer {
    const char* name;
    std::chrono::high_resolution_clock::time_point t0;
    explicit ScopedTimer(const char* n)
        : name(n), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[TIMING] " << name << " took " << ms << " ms\n";
    }
};

// ───── common types/helpers ────────────────────────────────────────────
struct Det { int cid; float score; float x1, y1, x2, y2; };
static inline float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }

// letterbox top/left: keep AR, pad right/bottom to (targetW,targetH)
static void letterbox_tl(const cv::Mat& src, cv::Mat& dst,
                         int targetW, int targetH, float& ratio) {
    int ow = src.cols, oh = src.rows;
    ratio = (std::min)(targetW / (float)ow, targetH / (float)oh);
    int nw = int(ow * ratio), nh = int(oh * ratio);
    cv::resize(src, dst, {nw, nh});
    cv::copyMakeBorder(dst, dst, 0, targetH - nh, 0, targetW - nw,
                       cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
}

// YOLOX decode (single head tensor layout: [N, sum(hw), 5+num_cls])
static std::vector<Det> decodeYOLOX(const Tensor& T,
                                    int ow, int oh, int iw, int ih, float r,
                                    const std::vector<int>& strides,
                                    float confThresh = 0.5f) {
    const float* buf = T.data.data();
    int clsN = int(T.shape[2]) - 5;
    int64_t gid = 0;
    std::vector<Det> out; out.reserve(2048);

    for (int s : strides) {
        int fw = (iw + s - 1) / s, fh = (ih + s - 1) / s;
        for (int y = 0; y < fh; ++y) {
            for (int x = 0; x < fw; ++x, ++gid) {
                const float* b = buf + gid * (5 + clsN);
                float obj = sigm(b[4]); if (obj < confThresh) continue;

                float bestP = 0.f; int bestC = -1;
                for (int c = 0; c < clsN; ++c) {
                    float p = sigm(b[5 + c]);
                    if (p > bestP) { bestP = p; bestC = c; }
                }
                float score = obj * bestP; if (score < confThresh) continue;

                float cxp = (x + sigm(b[0])) * s;
                float cyp = (y + sigm(b[1])) * s;
                float wp  = std::exp(b[2]) * s;
                float hp  = std::exp(b[3]) * s;

                float x1 = (cxp - wp * 0.5f) / r;
                float y1 = (cyp - hp * 0.5f) / r;
                float x2 = (cxp + wp * 0.5f) / r;
                float y2 = (cyp + hp * 0.5f) / r;

                x1 = std::clamp(x1, 0.f, (float)ow);
                y1 = std::clamp(y1, 0.f, (float)oh);
                x2 = std::clamp(x2, 0.f, (float)ow);
                y2 = std::clamp(y2, 0.f, (float)oh);

                out.push_back({bestC, score, x1, y1, x2, y2});
            }
        }
    }
    return out;
}

// Fast: BGR Mat -> RGB float -> (optional) (x-mean)/std -> CHW vector
static void makeBlobCHW(const cv::Mat& bgr, float scale,
                        const float mean[3], const float stdv[3],
                        std::vector<float>& out) {
    cv::Mat rgb; cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f32; rgb.convertTo(f32, CV_32F, scale);   // scale=1.f or 1/255.f
    if (mean && stdv) {
        cv::subtract(f32, cv::Scalar(mean[0], mean[1], mean[2]), f32);
        cv::divide(  f32, cv::Scalar(stdv[0], stdv[1], stdv[2]), f32);
    }
    std::vector<cv::Mat> ch(3); cv::split(f32, ch);   // HxWx3 -> 3x(HxW)
    const int N = (int)ch[0].total();
    out.resize(3 * N);
    std::memcpy(out.data() + 0*N, ch[0].ptr<float>(), N * sizeof(float));
    std::memcpy(out.data() + 1*N, ch[1].ptr<float>(), N * sizeof(float));
    std::memcpy(out.data() + 2*N, ch[2].ptr<float>(), N * sizeof(float));
}

static inline int bucket_width(int w) {
    static const int buckets[] = {96,128,160,192,224,256,288,320,352,384,448,512};
    int best = buckets[0], best_d = std::abs(w - buckets[0]);
    for (int b : buckets) { int d = std::abs(w - b); if (d < best_d) { best=b; best_d=d; } }
    return best;
}

// ── DB helpers: order/unclip/warp ──────────────────────────────────────
static std::array<cv::Point2f,4> orderQuad(std::array<cv::Point2f,4> o) {
    std::sort(o.begin(), o.end(),
        [](const cv::Point2f& a, const cv::Point2f& b){ return (a.x<b.x) || (a.x==b.x && a.y<b.y); });
    cv::Point2f tl = (o[0].y < o[1].y) ? o[0] : o[1];
    cv::Point2f bl = (o[0].y < o[1].y) ? o[1] : o[0];
    cv::Point2f tr = (o[2].y < o[3].y) ? o[2] : o[3];
    cv::Point2f br = (o[2].y < o[3].y) ? o[3] : o[2];
    return {tl,tr,br,bl};
}
static std::array<cv::Point2f,4> rectPts(const cv::RotatedRect& rr) {
    cv::Point2f p[4]; rr.points(p); return orderQuad({p[0],p[1],p[2],p[3]});
}
static std::array<cv::Point2f,4> unclipRect(const cv::RotatedRect& rr, float ratio) {
    cv::RotatedRect e = rr;
    e.size.width  = rr.size.width  * (1.f + ratio*0.5f);
    e.size.height = rr.size.height * (1.f + ratio*0.5f);
    return rectPts(e);
}
static cv::Mat warpQuadToSize(const cv::Mat& img,
                              const std::array<cv::Point2f,4>& poly,
                              int w, int h) {
    auto ord = orderQuad(poly);
    std::array<cv::Point2f,4> dst = { cv::Point2f(0,0), cv::Point2f((float)w,0),
                                      cv::Point2f((float)w,(float)h), cv::Point2f(0,(float)h) };
    cv::Mat M = cv::getPerspectiveTransform(ord.data(), dst.data());
    cv::Mat out;
    cv::warpPerspective(img, out, M, cv::Size(w,h), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return out;
}

// ── dot-peen enhancement (CLAHE → tophat → blur → adaptive bin) ───────
static cv::Mat enhance_dotpeen(const cv::Mat& src) {
    cv::Mat g; if (src.channels()==3) cv::cvtColor(src, g, cv::COLOR_BGR2GRAY); else g = src;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
    clahe->apply(g, g);
    int k = (int)(0.02 * (std::min)(src.cols, src.rows)); if (k<3) k=3; if ((k&1)==0) ++k;
    cv::Mat se = cv::getStructuringElement(cv::MORPH_RECT, {k,k});
    cv::morphologyEx(g, g, cv::MORPH_TOPHAT, se);
    cv::GaussianBlur(g, g, {3,3}, 0);
    cv::Mat bin;
    int block = (std::max)(15, k|1); // odd
    cv::adaptiveThreshold(g, bin, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, block, -6);
    cv::Mat out; cv::cvtColor(bin, out, cv::COLOR_GRAY2BGR);
    return out;
}

// ── IoU helper (for DM-like skipping) ──────────────────────────────────
static float IoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = (std::max)(a.x,b.x), y1=(std::max)(a.y,b.y);
    int x2 = (std::min)(a.x+a.width, b.x+b.width);
    int y2 = (std::min)(a.y+a.height,b.y+b.height);
    int w = (std::max)(0, x2-x1), h=(std::max)(0, y2-y1);
    float inter = float(w*h), uni = float(a.area()+b.area()-w*h);
    return uni>0 ? inter/uni : 0.f;
}

// ── CTC decode + avg confidence (blank=0). Filters to ASCII set. ──────
static std::pair<std::string,float>
ctcDecodeAutoWithConf(const Tensor& Yr, const std::vector<std::string>& dict) {
    if (Yr.shape.size()!=3 || Yr.shape[0]!=1) return {"",0.f};
    const float* D = Yr.data.data();
    int d1=(int)Yr.shape[1], d2=(int)Yr.shape[2];
    int T, C; bool is_T_C;
    if (d2 > d1) { T=d1; C=d2; is_T_C=true; } else { T=d2; C=d1; is_T_C=false; }

    static bool warned = false;
    if (!warned && (int)dict.size() != C-1) {
        std::cerr << "[WARN] REC dict size (" << dict.size()
                  << ") != model classes-1 (" << (C-1)
                  << "). Use the correct keys file for this model (e.g., multilingual 'ppocr_keys_v1.txt').\n";
        warned = true;
    }

    auto softmax_row = [&](const float* row)->std::vector<float>{
        float m = row[0]; for(int c=1;c<C;++c) m = (std::max)(m,row[c]);
        double sum=0.0; std::vector<float> p(C);
        for(int c=0;c<C;++c){ double e=std::exp(double(row[c]-m)); sum+=e; p[c]=(float)e; }
        for(int c=0;c<C;++c) p[c] = (float)(p[c]/sum);
        return p;
    };

    std::string out; int prev=-1; double conf_sum=0.0; int conf_cnt=0;
    for (int t=0;t<T;++t){
        std::vector<float> row(C);
        if (is_T_C) std::memcpy(row.data(), D + t*C, C*sizeof(float));
        else        for(int c=0;c<C;++c) row[c] = D[c*T + t];

        auto p = softmax_row(row.data());
        int k = int(std::max_element(p.begin(), p.end()) - p.begin());
        if (k!=prev && k>0 && k <= (int)dict.size()) {
            out += dict[k-1];
            conf_sum += p[k];
            ++conf_cnt;
        }
        prev = k;
    }

    std::string filtered;
    for (unsigned char ch : out)
        if ((ch>='0' && ch<='9') || (ch>='A'&&ch<='Z') || (ch>='a'&&ch<='z')
            || ch==' '||ch=='.'||ch=='-'||ch=='/'||ch==':') filtered.push_back((char)ch);

    float avg = conf_cnt>0 ? float(conf_sum/conf_cnt) : 0.f;
    return {filtered, avg};
}

// ── dict loader ────────────────────────────────────────────────────────
static bool load_keys_file(const std::string& path, std::vector<std::string>& dict) {
    std::ifstream ifs(path);
    if (!ifs) return false;
    std::string line;
    while (std::getline(ifs, line)) dict.push_back(line);
    return !dict.empty();
}

// ───── main ────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); SetConsoleCP(CP_UTF8);
#endif

    if (argc < 3){
        std::cerr << "Usage (legacy):\n"
                  << "  " << argv[0] << " model.onnx image.jpg\n"
                  << "  " << argv[0] << " det.onnx rec.onnx image.jpg [keys.txt]\n"
                  << "Usage (preferred harness):\n"
                  << "  " << argv[0] << " det.onnx rec.onnx --images list.txt --runs 50 --warmup 5 --threads 1 --csv out.csv --labels labels.csv --keys ppocr_keys_v1.txt\n";
        return 1;
    }

    Args A = parse_args(argc, argv);

    // If user passed a single image (no --images) and didn’t set runs/warmup,
    // default to a single pass for legacy behavior.
    if (!A.listPath.size() && A.singleImage.size() && !A.runs_set && !A.warmup_set){
        A.runs = 1;
        A.warmup = 0;
    }


    // ---- Detect-only (legacy): app model.onnx image.jpg ----
    if (A.recPath.empty() && !A.detPath.empty() && !A.singleImage.empty()){
        const char* imgPath = A.singleImage.c_str();
        cv::Mat orig = cv::imread(imgPath);
        if (orig.empty()) { std::cerr << "Cannot load " << imgPath << "\n"; return 1; }
        const int ow = orig.cols, oh = orig.rows;

        ScopedTimer totalDetect("DETECT pipeline: total");
        ONNXInference engine(A.detPath);

        // Input size (handle dynamic)
        auto ish = engine.inputShapes()[0];
        int iw = (int)ish[3], ih = (int)ish[2];
        if (iw<=0 || ih<=0) {
            iw = (std::min)(1024, ((ow + 31) / 32) * 32);
            ih = (std::min)( 640, ((oh + 31) / 32) * 32);
            engine.fixDynamicHW(ih, iw);
            std::cout << "[DEBUG] dynamic-shape using " << iw << "x" << ih << "\n";
        }

        // Is it PP-OCR det or YOLOX?
        auto osh = engine.outputShapes()[0];
        bool textDet = (osh.size()==4 && osh[1]==1);
        std::cout << "[DEBUG] " << (textDet ? "PP-OCRv5 text-det" : "YOLOX box-det") << " selected\n";

        // Preprocess
        float r = 1.f; cv::Mat inp;
        { ScopedTimer t("detect: preprocess"); letterbox_tl(orig, inp, iw, ih, r); }

        std::vector<float> blob;
        if (textDet) {
            const float MEAN[3]={0.485f,0.456f,0.406f}, STD[3]={0.229f,0.224f,0.225f};
            makeBlobCHW(inp, 1.f/255.f, MEAN, STD, blob);
        } else {
            makeBlobCHW(inp, 1.f, nullptr, nullptr, blob);
        }

        (void)engine.infer({blob}); // warm-up

        // Run
        std::vector<Tensor> outs;
        { ScopedTimer t("detect: ORT run"); outs = engine.infer({blob}); }
        const Tensor& Y = outs[0];

        // Decode
        std::vector<Det> dets;
        {
            ScopedTimer t("detect: decode");
            if (textDet) {
                int H=(int)Y.shape[2], W=(int)Y.shape[3];
                cv::Mat prob(H, W, CV_32F, (float*)Y.data.data());
                cv::Mat pr; cv::resize(prob, pr, {ow, oh});

                // threshold + morphology
                cv::Mat mask;
                cv::threshold(pr, mask, DB_BIN_THRESH, 255, cv::THRESH_BINARY);
                mask.convertTo(mask, CV_8U);
                cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                                 cv::getStructuringElement(cv::MORPH_RECT,{5,3}));
                cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                                 cv::getStructuringElement(cv::MORPH_RECT,{3,3}));

                // DM-like region to skip later
                std::vector<cv::Rect> dm_like;
                {
                    std::vector<std::vector<cv::Point>> tmp;
                    cv::findContours(mask, tmp, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
                    for (auto& c : tmp) {
                        cv::Rect b = cv::boundingRect(c);
                        float ar = float(b.width)/b.height;
                        if (ar>0.85f && ar<1.15f && b.area()>15000 && b.area()<300000) {
                            double fill = cv::mean(mask(b))[0]/255.0; // white ratio
                            if (fill>0.18 && fill<0.60) dm_like.push_back(b);
                        }
                    }
                }

                std::vector<std::vector<cv::Point>> cs;
                cv::findContours(mask, cs, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
                for (auto& c : cs) {
                    if (cv::contourArea(c) < DB_MIN_AREA) continue;
                    cv::RotatedRect rr = cv::minAreaRect(c);
                    auto poly = USE_DB_ROTATED_BOX ? unclipRect(rr, DB_UNCLIP_RATIO)
                                                   : rectPts(rr);
                    cv::Rect b = cv::boundingRect(std::vector<cv::Point2f>(poly.begin(), poly.end()));
                    if (b.area() < DB_MIN_AREA) continue;
                    bool hit_dm = std::any_of(dm_like.begin(), dm_like.end(),
                                              [&](const cv::Rect& r2){ return IoU(r2,b) > 0.30f; });
                    if (hit_dm) continue;

                    dets.push_back({0,1.f,(float)b.x,(float)b.y,(float)(b.x+b.width),(float)(b.y+b.height)});
                }
                std::cout << "[DEBUG] text boxes = " << dets.size() << "\n";
            } else {
                float rdet = r;
                dets = decodeYOLOX(Y, ow, oh, iw, ih, rdet, {8,16,32}, 0.5f);
                std::vector<cv::Rect> boxes; std::vector<float> scores;
                boxes.reserve(dets.size()); scores.reserve(dets.size());
                for (auto& d : dets) { boxes.emplace_back((int)d.x1,(int)d.y1,(int)(d.x2-d.x1),(int)(d.y2-d.y1)); scores.push_back(d.score); }
                std::vector<int> keep; cv::dnn::NMSBoxes(boxes, scores, 0.5f, 0.6f, keep);
                std::vector<Det> tmp; tmp.reserve(keep.size()); for (int k:keep) tmp.push_back(dets[k]); dets.swap(tmp);
                std::cout << "[DEBUG] kept = " << dets.size() << "\n";
            }
        }

        // Draw + save
        {
            ScopedTimer t("detect: draw+save");
            for (auto& d : dets) {
                cv::rectangle(orig, {int(d.x1),int(d.y1)}, {int(d.x2),int(d.y2)}, {0,255,0}, 2);
            }
            cv::imwrite("annotated.jpg", orig);
            std::cout << "[DEBUG] Saved annotated.jpg\n";
        }
        return 0;
    }

    // ---- FULL OCR HARNESS ----
    if (A.keysPath.empty()) A.keysPath = "ppocr_keys_v1.txt";
    if (A.singleImage.empty() && A.listPath.empty()){
        std::cerr<<"[ERROR] Provide a single image or --images list.txt\n";
        return 2;
    }

    // Load image list
    std::vector<std::string> images;
    if (!A.listPath.empty()){
        std::ifstream in(A.listPath);
        if (!in){ std::cerr<<"[ERROR] cannot open images list "<<A.listPath<<"\n"; return 2; }
        for(std::string line; std::getline(in,line);) if(!line.empty()) images.push_back(line);
    } else {
        images.push_back(A.singleImage);
    }

    // Engines
    ONNXInference detEngine(A.detPath), recEngine(A.recPath);
    // Optional: if your wrapper supports it, set threads here.
    // detEngine.setThreads(A.threads);
    // recEngine.setThreads(A.threads);

    // Load dict
    std::vector<std::string> dict;
    if (!A.keysPath.empty() && load_keys_file(A.keysPath, dict)) {
        std::cout << "[DEBUG] Loaded keys: " << A.keysPath << " (" << dict.size() << " entries)\n";
    } else if (load_keys_file("ppocrv5_dict.txt", dict)) {
        std::cout << "[DEBUG] Loaded keys: ppocrv5_dict.txt (" << dict.size() << " entries)\n";
    } else {
        std::cout << "[WARN] No keys file found. Falling back to 0-9A-Za-z.\n";
        for (char c='0'; c<='9'; ++c) dict.emplace_back(1,c);
        for (char c='A'; c<='Z'; ++c) dict.emplace_back(1,c);
        for (char c='a'; c<='z'; ++c) dict.emplace_back(1,c);
    }

    GroundTruth gt(A.labelsPath);
    CSV csv(A.csvPath);

    // ------------ WARMUP ------------
    for (int w=0; w<A.warmup; ++w){
        for (auto& imgPath: images){
            cv::Mat orig = cv::imread(imgPath); if (orig.empty()) continue;
            const int ow = orig.cols, oh = orig.rows;

            // DET size
            auto dIsh = detEngine.inputShapes()[0];
            int diw = (int)dIsh[3], dih = (int)dIsh[2];
            if (diw<=0 || dih<=0) {
                diw = (std::min)(1024, ((ow + 31) / 32) * 32);
                dih = (std::min)( 640, ((oh + 31) / 32) * 32);
                detEngine.fixDynamicHW(dih, diw);
            }
            float r = 1.f; cv::Mat inp; letterbox_tl(orig, inp, diw, dih, r);
            std::vector<float> detBlob;
            const float MEAN[3]={0.485f,0.456f,0.406f}, STD[3]={0.229f,0.224f,0.225f};
            makeBlobCHW(inp, 1.f/255.f, MEAN, STD, detBlob);
            (void)detEngine.infer({detBlob});

            // REC pre-warm some widths
            std::vector<int> warm = {128,192,256,320,384,448,512};
            int rcH = (int)recEngine.inputShapes()[0][2];
            for (int bw : warm) {
                if (bw > REC_MAX_W) continue;
                recEngine.fixExactInputShape(0, {1,3,rcH,bw});
                std::vector<float> dummy(size_t(1)*3*rcH*bw, 0.f);
                (void)recEngine.infer({dummy});
            }
        }
    }

    // ------------ MEASURED RUNS ------------
    for (int r=0; r<A.runs; ++r){
        for (auto& imgPath: images){
            cv::Mat orig = cv::imread(imgPath);
            if (orig.empty()) { std::cerr<<"Cannot load "<<imgPath<<"\n"; continue; }
            const int ow = orig.cols, oh = orig.rows;

            double det_pre_ms=0, det_run_ms=0, det_dec_ms=0, rec_total_ms=0;
            auto t_mark = [](){ return std::chrono::high_resolution_clock::now(); };
            auto ms_since = [&](auto t0){
                return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
            };
            auto t_total_start = t_mark();

            // ---- DETECT: preprocess ----
            float rscale = 1.f; cv::Mat inp;
            // input sizing (dynamic)
            auto dIsh = detEngine.inputShapes()[0];
            int diw = (int)dIsh[3], dih = (int)dIsh[2];
            if (diw<=0 || dih<=0) {
                diw = (std::min)(1024, ((ow + 31) / 32) * 32);
                dih = (std::min)( 640, ((oh + 31) / 32) * 32);
                detEngine.fixDynamicHW(dih, diw);
            }
            {
                auto __t = t_mark();
                { ScopedTimer t("ocr: det preprocess"); letterbox_tl(orig, inp, diw, dih, rscale); }
                det_pre_ms = ms_since(__t);
            }

            std::vector<float> detBlob;
            {
                const float MEAN[3]={0.485f,0.456f,0.406f}, STD[3]={0.229f,0.224f,0.225f};
                makeBlobCHW(inp, 1.f/255.f, MEAN, STD, detBlob);
            }
            detEngine.infer({detBlob}); // warmup call (cheap)

            // ---- DETECT: ORT run ----
            std::vector<Tensor> detOut;
            {
                auto __t = t_mark();
                { ScopedTimer t("ocr: det ORT run"); detOut = detEngine.infer({detBlob}); }
                det_run_ms = ms_since(__t);
            }
            const Tensor& Yd = detOut[0];

            // ---- DETECT: decode ----
            std::vector<std::array<cv::Point2f,4>> quads;
            std::vector<Det> boxes;
            {
                auto __t = t_mark();
                ScopedTimer t("ocr: det decode");
                int H=(int)Yd.shape[2], W=(int)Yd.shape[3];
                std::cerr << "[DEBUG] heatmap dims H=" << H << " W=" << W << "\n";
                if (H<=0 || W<=0) { std::cerr << "[ERROR] Empty heatmap.\n"; continue; }

                cv::Mat prob(H,W,CV_32F,(float*)Yd.data.data());
                cv::Mat pr; cv::resize(prob, pr, {ow, oh});

                cv::Mat mask;
                cv::threshold(pr, mask, DB_BIN_THRESH, 255, cv::THRESH_BINARY);
                mask.convertTo(mask, CV_8U);
                cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                                 cv::getStructuringElement(cv::MORPH_RECT,{5,3}));
                cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                                 cv::getStructuringElement(cv::MORPH_RECT,{3,3}));

                // DM-like blacklist
                std::vector<cv::Rect> dm_like;
                {
                    std::vector<std::vector<cv::Point>> tmp;
                    cv::findContours(mask, tmp, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
                    for (auto& c : tmp) {
                        cv::Rect b = cv::boundingRect(c);
                        float ar = float(b.width)/b.height;
                        if (ar>0.85f && ar<1.15f && b.area()>15000 && b.area()<300000) {
                            double fill = cv::mean(mask(b))[0]/255.0;
                            if (fill>0.18 && fill<0.60) dm_like.push_back(b);
                        }
                    }
                }

                std::vector<std::vector<cv::Point>> cs;
                cv::findContours(mask, cs, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
                for (auto& c : cs) {
                    if (cv::contourArea(c) < DB_MIN_AREA) continue;
                    cv::RotatedRect rr = cv::minAreaRect(c);
                    auto poly = USE_DB_ROTATED_BOX ? unclipRect(rr, DB_UNCLIP_RATIO) : rectPts(rr);
                    cv::Rect b = cv::boundingRect(std::vector<cv::Point2f>(poly.begin(), poly.end()));
                    if (b.area() < DB_MIN_AREA) continue;
                    bool hit_dm = std::any_of(dm_like.begin(), dm_like.end(),
                                              [&](const cv::Rect& r2){ return IoU(r2,b) > 0.30f; });
                    if (hit_dm) continue;

                    quads.push_back(poly);
                    boxes.push_back({0,1.f,(float)b.x,(float)b.y,(float)(b.x+b.width),(float)(b.y+b.height)});
                }
                det_dec_ms = ms_since(__t);
            }
            std::cout << "[DEBUG] text boxes = " << boxes.size() << "\n";

            // ---- Load/confirm dict already done above ----

            // ---- REC: pre-warm some widths (one-time could be moved, but ok) ----
            {
                std::vector<int> warm = {128,192,256,320,384,448,512};
                int rcH = (int)recEngine.inputShapes()[0][2];
                for (int bw : warm) {
                    if (bw > REC_MAX_W) continue;
                    recEngine.fixExactInputShape(0, {1,3,rcH,bw});
                    std::vector<float> dummy(size_t(1)*3*rcH*bw, 0.f);
                    (void)recEngine.infer({dummy});
                }
            }

            // ---- REC: per box ----
            auto rIsh = recEngine.inputShapes()[0];
            const int rcH = (int)rIsh[2];

            std::vector<std::string> per_box_text; per_box_text.reserve(boxes.size());
            auto __t_rec_all = t_mark();

            for (size_t i=0;i<boxes.size();++i) {
                const auto& d = boxes[i];
                cv::Rect rr{ (int)d.x1, (int)d.y1, (int)(d.x2-d.x1), (int)(d.y2-d.y1) };
                rr &= cv::Rect(0,0,orig.cols,orig.rows);
                if (rr.width<=0 || rr.height<=0) continue;

                int recW; cv::Mat baseCrop;
                {
                    ScopedTimer tp("ocr: rec preprocess (per box)");
                    recW = int(float(rr.width) * rcH / (std::max)(1, rr.height));
                    recW = (std::min)(REC_MAX_W, bucket_width(recW));
                    if (recW < 32) recW = 32;

                    recEngine.fixExactInputShape(0, {1, 3, rcH, recW});

                    if (USE_PERSPECTIVE_CROP && i < quads.size()) {
                        baseCrop = warpQuadToSize(orig, quads[i], recW, rcH);
                    } else {
                        cv::resize(orig(rr), baseCrop, cv::Size(recW, rcH));
                    }
                }

                cv::Mat enhanced;
                if (APPLY_DOTPEEN_ENH) enhanced = enhance_dotpeen(baseCrop);

                auto run_rec_try = [&](const cv::Mat& in, bool use_ppocr_norm,
                                       bool to_gray, bool invert)->std::pair<std::string,float>
                {
                    cv::Mat img = in.clone();

                    if (to_gray) {
                        if (img.channels() == 3) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
                        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
                    }
                    if (invert) cv::bitwise_not(img, img);

                    std::vector<float> blob; blob.reserve(size_t(3) * rcH * recW);
                    if (use_ppocr_norm) {
                        const float M[3] = {0.5f, 0.5f, 0.5f};
                        const float S[3] = {0.5f, 0.5f, 0.5f};
                        makeBlobCHW(img, 1.f/255.f, M, S, blob);
                    } else {
                        makeBlobCHW(img, 1.f/255.f, nullptr, nullptr, blob);
                    }

                    auto recOut = recEngine.infer({blob});
                    return ctcDecodeAutoWithConf(recOut[0], dict);
                };

                std::pair<std::string,float> best = {"", 0.f};
                best = run_rec_try(baseCrop, REC_USE_PPOCR_NORM, false, false);  // 1) default
                { auto r = run_rec_try(baseCrop, false, false, false); if (r.second > best.second) best = r; }        // 2) no norm
                { auto r = run_rec_try(baseCrop, false, true,  false); if (r.second > best.second) best = r; }        // 3) gray
                { auto r = run_rec_try(baseCrop, false, true,  true ); if (r.second > best.second) best = r; }        // 4) gray+invert
                if (APPLY_DOTPEEN_ENH && !enhanced.empty()){
                    auto rtry = run_rec_try(enhanced, false, false, false);
                    if (rtry.second > best.second) best = rtry;                                                      // 5) enhanced
                }

                std::string txt = (best.second >= CTC_REJECT_AVGCONF) ? best.first : "";
                std::cout << "[REC] \"" << txt << "\"  (avgconf=" << best.second << ")\n";
                if (!txt.empty()) per_box_text.push_back(txt);

                // optional draw (disabled for harness speed)
                // cv::rectangle(orig, rr, {255,0,0}, 2);
            }

            rec_total_ms = ms_since(__t_rec_all);

            // Concat text
            std::string text_concat;
            for (size_t k=0;k<per_box_text.size();++k){
                if (k) text_concat += " ";
                text_concat += per_box_text[k];
            }

            double total_ms = std::chrono::duration<double,std::milli>(
                    std::chrono::high_resolution_clock::now() - t_total_start).count();

            // ---------- Accuracy (optional) ----------
            std::string truth = gt.get(imgPath);
            double cer=-1, wer=-1;
            if (!truth.empty()){
                int d_chars = lev(text_concat, truth);
                cer = (truth.empty()?0.0 : (double)d_chars / (double)truth.size());
                auto H = split_words(text_concat), R = split_words(truth);
                auto join = [](const std::vector<std::string>& v){
                    std::string o; for(size_t i=0;i<v.size();++i){ if(i) o.push_back('\x1F'); o+=v[i]; } return o; };
                int d_words = lev(join(H), join(R));
                wer = (R.empty()?0.0 : (double)d_words / (double)R.size());
            }

            // ---------- CSV ----------
            csv.line(now_us(), imgPath, r, 0, A.threads,
                     det_pre_ms, det_run_ms, det_dec_ms,
                     rec_total_ms, (int)boxes.size(), total_ms,
                     text_concat, cer, wer);
        }
    }

    std::cout << "[DONE] Wrote results to " << A.csvPath << "\n";
    return 0;
}
