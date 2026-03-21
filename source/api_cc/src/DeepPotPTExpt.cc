// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepPotPTExpt.h"

#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>

#include "SimulationRegion.h"
#include "common.h"
#include "device.h"
#include "errors.h"
#include "neighbor_list.h"

// Minimal JSON value parser for reading metadata from .pt2 archives.
// Supports: strings, numbers, booleans, arrays, objects.
// This avoids adding a dependency on nlohmann/json for the api_cc library.
namespace {

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
  enum Type { Null, Bool, Number, String, Array, Object };
  Type type = Null;
  bool bool_val = false;
  double num_val = 0.0;
  std::string str_val;
  JsonArray arr_val;
  JsonObject obj_val;

  std::string as_string() const { return str_val; }
  double as_double() const { return num_val; }
  int as_int() const { return static_cast<int>(num_val); }
  bool as_bool() const { return bool_val; }
  const JsonArray& as_array() const { return arr_val; }
  const JsonObject& as_object() const { return obj_val; }
  const JsonValue& operator[](const std::string& key) const {
    return obj_val.at(key);
  }
  const JsonValue& operator[](size_t idx) const { return arr_val.at(idx); }
  bool has(const std::string& key) const {
    return obj_val.find(key) != obj_val.end();
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& s) : s_(s), pos_(0) {}
  JsonValue parse() {
    skip_ws();
    auto val = parse_value();
    return val;
  }

 private:
  const std::string& s_;
  size_t pos_;

  char peek() const { return pos_ < s_.size() ? s_[pos_] : '\0'; }
  char get() {
    if (pos_ >= s_.size()) {
      throw std::runtime_error("JSON parse error: unexpected end of input");
    }
    return s_[pos_++];
  }
  void skip_ws() {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' ||
                                s_[pos_] == '\n' || s_[pos_] == '\r')) {
      ++pos_;
    }
  }

  JsonValue parse_value() {
    skip_ws();
    char c = peek();
    if (c == '"') {
      return parse_string_val();
    } else if (c == '{') {
      return parse_object();
    } else if (c == '[') {
      return parse_array();
    } else if (c == 't' || c == 'f') {
      return parse_bool();
    } else if (c == 'n') {
      return parse_null();
    } else {
      return parse_number();
    }
  }

  std::string parse_string_raw() {
    get();  // consume '"'
    std::string result;
    while (pos_ < s_.size() && peek() != '"') {
      if (peek() == '\\') {
        get();
        char esc = get();
        switch (esc) {
          case '"':
            result += '"';
            break;
          case '\\':
            result += '\\';
            break;
          case '/':
            result += '/';
            break;
          case 'n':
            result += '\n';
            break;
          case 't':
            result += '\t';
            break;
          case 'r':
            result += '\r';
            break;
          default:
            result += esc;
            break;
        }
      } else {
        result += get();
      }
    }
    get();  // consume closing '"'
    return result;
  }

  JsonValue parse_string_val() {
    JsonValue v;
    v.type = JsonValue::String;
    v.str_val = parse_string_raw();
    return v;
  }

  JsonValue parse_number() {
    size_t start = pos_;
    if (peek() == '-') {
      get();
    }
    while (pos_ < s_.size() &&
           (std::isdigit(s_[pos_]) || s_[pos_] == '.' || s_[pos_] == 'e' ||
            s_[pos_] == 'E' || s_[pos_] == '+' || s_[pos_] == '-')) {
      // handle sign only if after e/E
      if ((s_[pos_] == '+' || s_[pos_] == '-') && pos_ > start &&
          s_[pos_ - 1] != 'e' && s_[pos_ - 1] != 'E') {
        break;
      }
      ++pos_;
    }
    JsonValue v;
    v.type = JsonValue::Number;
    try {
      v.num_val = std::stod(s_.substr(start, pos_ - start));
    } catch (const std::exception& e) {
      throw std::runtime_error("JSON parse error: invalid number at position " +
                               std::to_string(start));
    }
    return v;
  }

  JsonValue parse_bool() {
    JsonValue v;
    v.type = JsonValue::Bool;
    if (s_.substr(pos_, 4) == "true") {
      v.bool_val = true;
      pos_ += 4;
    } else if (s_.substr(pos_, 5) == "false") {
      v.bool_val = false;
      pos_ += 5;
    } else {
      throw std::runtime_error(
          "JSON parse error: expected 'true' or 'false' at position " +
          std::to_string(pos_));
    }
    return v;
  }

  JsonValue parse_null() {
    if (s_.substr(pos_, 4) != "null") {
      throw std::runtime_error(
          "JSON parse error: expected 'null' at position " +
          std::to_string(pos_));
    }
    pos_ += 4;
    return JsonValue();
  }

  JsonValue parse_array() {
    get();  // consume '['
    JsonValue v;
    v.type = JsonValue::Array;
    skip_ws();
    if (peek() == ']') {
      get();
      return v;
    }
    while (true) {
      v.arr_val.push_back(parse_value());
      skip_ws();
      if (peek() == ',') {
        get();
      } else {
        break;
      }
    }
    skip_ws();
    get();  // consume ']'
    return v;
  }

  JsonValue parse_object() {
    get();  // consume '{'
    JsonValue v;
    v.type = JsonValue::Object;
    skip_ws();
    if (peek() == '}') {
      get();
      return v;
    }
    while (true) {
      skip_ws();
      std::string key = parse_string_raw();
      skip_ws();
      get();  // consume ':'
      v.obj_val[key] = parse_value();
      skip_ws();
      if (peek() == ',') {
        get();
      } else {
        break;
      }
    }
    skip_ws();
    get();  // consume '}'
    return v;
  }
};

JsonValue parse_json(const std::string& s) {
  JsonParser parser(s);
  return parser.parse();
}

// Read a file from a ZIP archive using caffe2::serialize::PyTorchStreamReader.
// We avoid depending on caffe2 headers by using a simpler approach:
// just read the file directly as a ZIP file.
std::string read_zip_entry(const std::string& zip_path,
                           const std::string& entry_name) {
  // Use a simple approach: scan all possible prefixed names.
  // .pt2 files from AOTInductor store extra files at "extra/<name>"
  // within the ZIP archive.
  std::ifstream ifs(zip_path, std::ios::binary);
  if (!ifs.is_open()) {
    throw deepmd::deepmd_exception("Cannot open file: " + zip_path);
  }

  // Read entire file
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  ifs.close();

  // Simple ZIP central directory parser
  // Find End of Central Directory Record (EOCD)
  // EOCD signature: 0x06054b50
  // Minimum EOCD size is 22 bytes
  if (content.size() < 22) {
    throw deepmd::deepmd_exception(
        "File too small to be a valid ZIP archive: " + zip_path);
  }
  size_t eocd_pos = std::string::npos;
  for (int64_t i = static_cast<int64_t>(content.size()) - 22;
       i >= 0 && static_cast<size_t>(i) + 3 < content.size(); --i) {
    if (content[i] == 0x50 && content[i + 1] == 0x4b &&
        content[i + 2] == 0x05 && content[i + 3] == 0x06) {
      eocd_pos = static_cast<size_t>(i);
      break;
    }
  }
  if (eocd_pos == std::string::npos) {
    throw deepmd::deepmd_exception("Invalid ZIP file: " + zip_path);
  }

  // Parse EOCD to get central directory offset and size
  auto read_u16 = [&](size_t offset) -> uint16_t {
    return static_cast<uint16_t>(static_cast<unsigned char>(content[offset])) |
           (static_cast<uint16_t>(
                static_cast<unsigned char>(content[offset + 1]))
            << 8);
  };
  auto read_u32 = [&](size_t offset) -> uint32_t {
    return static_cast<uint32_t>(static_cast<unsigned char>(content[offset])) |
           (static_cast<uint32_t>(
                static_cast<unsigned char>(content[offset + 1]))
            << 8) |
           (static_cast<uint32_t>(
                static_cast<unsigned char>(content[offset + 2]))
            << 16) |
           (static_cast<uint32_t>(
                static_cast<unsigned char>(content[offset + 3]))
            << 24);
  };

  uint64_t num_entries = read_u16(eocd_pos + 10);
  uint64_t cd_offset = read_u32(eocd_pos + 16);

  // If this is a ZIP64 file, look for the ZIP64 EOCD locator
  if (cd_offset == 0xFFFFFFFF || num_entries == 0xFFFF) {
    // ZIP64 EOCD locator signature: 0x07064b50
    // It should be right before the EOCD (20 bytes)
    if (eocd_pos < 20) {
      throw deepmd::deepmd_exception(
          "Invalid ZIP64 file (truncated EOCD locator): " + zip_path);
    }
    size_t zip64_locator_pos = eocd_pos - 20;
    if (content[zip64_locator_pos] == 0x50 &&
        content[zip64_locator_pos + 1] == 0x4b &&
        content[zip64_locator_pos + 2] == 0x06 &&
        content[zip64_locator_pos + 3] == 0x07) {
      // Read ZIP64 EOCD offset from locator
      uint64_t zip64_eocd_offset = 0;
      for (int b = 0; b < 8; ++b) {
        zip64_eocd_offset |= static_cast<uint64_t>(static_cast<unsigned char>(
                                 content[zip64_locator_pos + 8 + b]))
                             << (8 * b);
      }
      // Parse ZIP64 EOCD
      // ZIP64 EOCD signature: 0x06064b50
      size_t z64_pos = static_cast<size_t>(zip64_eocd_offset);
      if (z64_pos + 56 > content.size()) {
        throw deepmd::deepmd_exception(
            "Invalid ZIP64 file (truncated EOCD record): " + zip_path);
      }
      // num entries at offset 32 (8 bytes in ZIP64)
      num_entries = 0;
      for (int b = 0; b < 8; ++b) {
        num_entries |= static_cast<uint64_t>(static_cast<unsigned char>(
                           content[z64_pos + 32 + b]))
                       << (8 * b);
      }
      // cd offset at offset 48 (8 bytes in ZIP64)
      cd_offset = 0;
      for (int b = 0; b < 8; ++b) {
        cd_offset |= static_cast<uint64_t>(
                         static_cast<unsigned char>(content[z64_pos + 48 + b]))
                     << (8 * b);
      }
    }
  }

  // Iterate central directory entries
  size_t pos = cd_offset;
  for (uint64_t i = 0; i < num_entries; ++i) {
    // Central directory entry signature: 0x02014b50
    if (pos + 46 > content.size()) {
      break;
    }
    uint16_t name_len = read_u16(pos + 28);
    uint16_t extra_len = read_u16(pos + 30);
    uint16_t comment_len = read_u16(pos + 32);
    uint32_t compressed_size_u32 = read_u32(pos + 20);
    uint32_t uncompressed_size_u32 = read_u32(pos + 24);
    uint32_t local_header_offset_u32 = read_u32(pos + 42);

    // Use 64-bit types so ZIP64 values are not truncated
    uint64_t compressed_size = compressed_size_u32;
    uint64_t uncompressed_size = uncompressed_size_u32;
    uint64_t local_header_offset = local_header_offset_u32;

    std::string name = content.substr(pos + 46, name_len);

    // Handle ZIP64 extra field for large files
    if (uncompressed_size_u32 == 0xFFFFFFFF ||
        local_header_offset_u32 == 0xFFFFFFFF) {
      // Parse ZIP64 extended information extra field
      size_t extra_pos = pos + 46 + name_len;
      size_t extra_end = extra_pos + extra_len;
      while (extra_pos + 4 <= extra_end) {
        uint16_t field_id = read_u16(extra_pos);
        uint16_t field_size = read_u16(extra_pos + 2);
        if (field_id == 0x0001) {  // ZIP64 extra field
          size_t field_data = extra_pos + 4;
          int offset_in_field = 0;
          if (uncompressed_size_u32 == 0xFFFFFFFF) {
            uncompressed_size = 0;
            for (int b = 0; b < 8; ++b) {
              uncompressed_size |=
                  static_cast<uint64_t>(static_cast<unsigned char>(
                      content[field_data + offset_in_field + b]))
                  << (8 * b);
            }
            offset_in_field += 8;
          }
          if (compressed_size_u32 == 0xFFFFFFFF) {
            compressed_size = 0;
            for (int b = 0; b < 8; ++b) {
              compressed_size |=
                  static_cast<uint64_t>(static_cast<unsigned char>(
                      content[field_data + offset_in_field + b]))
                  << (8 * b);
            }
            offset_in_field += 8;
          }
          if (local_header_offset_u32 == 0xFFFFFFFF) {
            local_header_offset = 0;
            for (int b = 0; b < 8; ++b) {
              local_header_offset |=
                  static_cast<uint64_t>(static_cast<unsigned char>(
                      content[field_data + offset_in_field + b]))
                  << (8 * b);
            }
          }
          break;
        }
        extra_pos += 4 + field_size;
      }
    }

    // Match exact name or suffix (handles archives with directory prefixes,
    // e.g. "model/extra/output_keys.json" matches "extra/output_keys.json")
    bool match = (name == entry_name);
    if (!match && name.size() > entry_name.size()) {
      size_t suffix_start = name.size() - entry_name.size();
      if (name[suffix_start - 1] == '/' &&
          name.substr(suffix_start) == entry_name) {
        match = true;
      }
    }
    if (match) {
      // Read from local file header
      uint16_t local_name_len = read_u16(local_header_offset + 26);
      uint16_t local_extra_len = read_u16(local_header_offset + 28);
      size_t data_offset =
          local_header_offset + 30 + local_name_len + local_extra_len;
      return content.substr(data_offset, uncompressed_size);
    }

    pos += 46 + name_len + extra_len + comment_len;
  }

  throw deepmd::deepmd_exception("Entry not found in ZIP: " + entry_name +
                                 " in " + zip_path);
}

}  // namespace

using namespace deepmd;

/**
 * @brief Convert a raw neighbor list to the sel-limited format expected by the
 *        pt_expt model.
 *
 * For non-mixed-type models (distinguish_types=true): the nlist has shape
 * (nframes, nloc, sum(sel)), where the first sel[0] entries are neighbors of
 * type 0, the next sel[1] are type 1, etc.  Within each type group neighbors
 * are sorted by distance (ascending).
 *
 * For mixed-type models (distinguish_types=false): all neighbors go into a
 * single group sorted by distance, truncated to sum(sel).
 *
 * Missing slots are filled with -1.
 *
 * @param[in]  raw_nlist    Raw neighbor list (nloc x variable-nnei).
 * @param[in]  coord_ext    Extended coordinates (nall x 3), flat.
 * @param[in]  atype_ext    Extended atom types (nall).
 * @param[in]  sel          Per-type neighbor selection counts.
 * @param[in]  nloc         Number of local atoms.
 * @param[in]  mixed_types  Whether the model uses mixed types
 *                          (distinguish_types=false).
 * @return Tensor of shape (1, nloc, sum(sel)), dtype int64.
 */
template <typename VALUETYPE>
torch::Tensor buildTypeSortedNlist(
    const std::vector<std::vector<int>>& raw_nlist,
    const std::vector<VALUETYPE>& coord_ext,
    const std::vector<int>& atype_ext,
    const std::vector<int>& sel,
    int nloc,
    bool mixed_types) {
  int nsel = 0;
  for (auto s : sel) {
    nsel += s;
  }
  int ntypes = sel.size();
  std::vector<int64_t> result(static_cast<size_t>(nloc) * nsel, -1);

  for (int ii = 0; ii < nloc; ++ii) {
    const auto& neighbors = raw_nlist[ii];
    VALUETYPE xi = coord_ext[ii * 3 + 0];
    VALUETYPE yi = coord_ext[ii * 3 + 1];
    VALUETYPE zi = coord_ext[ii * 3 + 2];
    int offset = ii * nsel;

    if (mixed_types) {
      // Mixed-type: all neighbors in one group, sort by distance
      std::vector<std::pair<VALUETYPE, int>> all_neighbors;
      for (int jj : neighbors) {
        if (jj < 0) {
          continue;
        }
        int jtype = atype_ext[jj];
        if (jtype < 0) {
          continue;  // skip invalid atoms
        }
        VALUETYPE dx = coord_ext[jj * 3 + 0] - xi;
        VALUETYPE dy = coord_ext[jj * 3 + 1] - yi;
        VALUETYPE dz = coord_ext[jj * 3 + 2] - zi;
        VALUETYPE rr = dx * dx + dy * dy + dz * dz;
        all_neighbors.emplace_back(rr, jj);
      }
      std::sort(all_neighbors.begin(), all_neighbors.end());
      int count = std::min(static_cast<int>(all_neighbors.size()), nsel);
      for (int kk = 0; kk < count; ++kk) {
        result[offset + kk] = all_neighbors[kk].second;
      }
    } else {
      // Non-mixed-type: group by type, sort each group
      std::vector<std::vector<std::pair<VALUETYPE, int>>> by_type(ntypes);
      for (int jj : neighbors) {
        if (jj < 0) {
          continue;
        }
        int jtype = atype_ext[jj];
        if (jtype < 0 || jtype >= ntypes) {
          continue;  // skip virtual/unknown type atoms
        }
        VALUETYPE dx = coord_ext[jj * 3 + 0] - xi;
        VALUETYPE dy = coord_ext[jj * 3 + 1] - yi;
        VALUETYPE dz = coord_ext[jj * 3 + 2] - zi;
        VALUETYPE rr = dx * dx + dy * dy + dz * dz;
        by_type[jtype].emplace_back(rr, jj);
      }
      int col = 0;
      for (int tt = 0; tt < ntypes; ++tt) {
        auto& group = by_type[tt];
        std::sort(group.begin(), group.end());
        int count = std::min(static_cast<int>(group.size()), sel[tt]);
        for (int kk = 0; kk < count; ++kk) {
          result[offset + col + kk] = group[kk].second;
        }
        col += sel[tt];
      }
    }
  }

  torch::Tensor tensor =
      torch::from_blob(result.data(), {1, nloc, nsel},
                       torch::TensorOptions().dtype(torch::kInt64))
          .clone();
  return tensor;
}

void DeepPotPTExpt::translate_error(std::function<void()> f) {
  try {
    f();
  } catch (const c10::Error& e) {
    throw deepmd::deepmd_exception(
        "DeePMD-kit PyTorch Exportable backend error: " +
        std::string(e.what()));
  } catch (const deepmd::deepmd_exception&) {
    throw;  // already a deepmd_exception, rethrow as-is
  } catch (const std::exception& e) {
    throw deepmd::deepmd_exception(
        "DeePMD-kit PyTorch Exportable backend error: " +
        std::string(e.what()));
  }
}

DeepPotPTExpt::DeepPotPTExpt() : inited(false) {}

DeepPotPTExpt::DeepPotPTExpt(const std::string& model,
                             const int& gpu_rank,
                             const std::string& file_content)
    : inited(false) {
  try {
    translate_error([&] { init(model, gpu_rank, file_content); });
  } catch (...) {
    throw;
  }
}

void DeepPotPTExpt::init(const std::string& model,
                         const int& gpu_rank,
                         const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }

  if (!file_content.empty()) {
    throw deepmd::deepmd_exception(
        "In-memory file_content loading is not supported for .pt2 models. "
        "Please provide a file path instead.");
  }

  int gpu_num = torch::cuda::device_count();
  gpu_id = (gpu_num > 0) ? (gpu_rank % gpu_num) : 0;
  gpu_enabled = torch::cuda::is_available();

  std::string device_str;
  if (!gpu_enabled) {
    device_str = "cpu";
    std::cout << "load model from: " << model << " to cpu" << std::endl;
  } else {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    DPErrcheck(DPSetDevice(gpu_id));
#endif
    device_str = "cuda:" + std::to_string(gpu_id);
    std::cout << "load model from: " << model << " to gpu " << gpu_id
              << std::endl;
  }

  // Read metadata from the .pt2 ZIP archive
  std::string metadata_json =
      read_zip_entry(model, "extra/model_def_script.json");
  std::string output_keys_json =
      read_zip_entry(model, "extra/output_keys.json");

  auto metadata = parse_json(metadata_json);
  rcut = metadata["rcut"].as_double();
  ntypes = static_cast<int>(metadata["type_map"].as_array().size());
  dfparam = metadata["dim_fparam"].as_int();
  daparam = metadata["dim_aparam"].as_int();
  mixed_types = metadata["mixed_types"].as_bool();
  aparam_nall = false;  // pt_expt models use nloc for aparam
  if (metadata.obj_val.count("has_default_fparam")) {
    has_default_fparam_ = metadata["has_default_fparam"].as_bool();
  } else {
    has_default_fparam_ = false;
  }

  type_map.clear();
  for (const auto& v : metadata["type_map"].as_array()) {
    type_map.push_back(v.as_string());
  }

  sel.clear();
  for (const auto& v : metadata["sel"].as_array()) {
    sel.push_back(v.as_int());
  }

  // Parse output keys
  auto keys_val = parse_json(output_keys_json);
  output_keys.clear();
  for (const auto& v : keys_val.as_array()) {
    output_keys.push_back(v.as_string());
  }

  // Load the AOTInductor model package
  loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
      model, "model", false, 1,
      gpu_enabled ? static_cast<c10::DeviceIndex>(gpu_id)
                  : static_cast<c10::DeviceIndex>(-1));

  int num_intra_nthreads, num_inter_nthreads;
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  if (num_inter_nthreads) {
    try {
      at::set_num_interop_threads(num_inter_nthreads);
    } catch (...) {
    }
  }
  if (num_intra_nthreads) {
    try {
      at::set_num_threads(num_intra_nthreads);
    } catch (...) {
    }
  }

  inited = true;
}

DeepPotPTExpt::~DeepPotPTExpt() {}

std::vector<torch::Tensor> DeepPotPTExpt::run_model(
    const torch::Tensor& coord,
    const torch::Tensor& atype,
    const torch::Tensor& nlist,
    const torch::Tensor& mapping,
    const torch::Tensor& fparam,
    const torch::Tensor& aparam) {
  // Only include fparam/aparam if the model was exported with them.
  // When fparam/aparam are None at export time, AOTInductor compiles
  // the model with fewer inputs (e.g. 4 instead of 6).
  std::vector<torch::Tensor> inputs = {coord, atype, nlist, mapping};
  if (dfparam > 0) {
    inputs.push_back(fparam);
  }
  if (daparam > 0) {
    inputs.push_back(aparam);
  }
  return loader->run(inputs);
}

void DeepPotPTExpt::extract_outputs(
    std::map<std::string, torch::Tensor>& output_map,
    const std::vector<torch::Tensor>& flat_outputs) {
  if (flat_outputs.size() != output_keys.size()) {
    throw deepmd::deepmd_exception(
        "Model returned " + std::to_string(flat_outputs.size()) +
        " outputs but expected " + std::to_string(output_keys.size()) +
        " (from output_keys.json)");
  }
  for (size_t i = 0; i < output_keys.size(); ++i) {
    output_map[output_keys[i]] = flat_outputs[i];
  }
}

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPTExpt::compute(ENERGYVTYPE& ener,
                            std::vector<VALUETYPE>& force,
                            std::vector<VALUETYPE>& virial,
                            std::vector<VALUETYPE>& atom_energy,
                            std::vector<VALUETYPE>& atom_virial,
                            const std::vector<VALUETYPE>& coord,
                            const std::vector<int>& atype,
                            const std::vector<VALUETYPE>& box,
                            const int nghost,
                            const InputNlist& lmp_list,
                            const int& ago,
                            const std::vector<VALUETYPE>& fparam,
                            const std::vector<VALUETYPE>& aparam,
                            const bool atomic) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  int natoms = atype.size();
  // Always use float64 for model inputs — the .pt2 model is compiled with
  // float64 and AOTInductor does not auto-cast.  We only cast outputs back
  // to VALUETYPE at the end.
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    floatType = torch::kFloat32;
  }
  auto int_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

  // Select real atoms (filter NULL-type atoms)
  std::vector<VALUETYPE> dcoord, dforce, aparam_, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, 1, daparam, nall, aparam_nall);
  int nloc = nall_real - nghost_real;
  int nframes = 1;

  // Convert coord to float64 for model input
  // NOTE: must .clone() because from_blob does not copy data, and the local
  // vectors would go out of scope before run_model completes.
  std::vector<double> coord_d(dcoord.begin(), dcoord.end());
  at::Tensor coord_Tensor =
      torch::from_blob(coord_d.data(), {1, nall_real, 3}, options)
          .clone()
          .to(device);
  std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_option)
          .clone()
          .to(device);

  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list, nall - nghost);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();
  }
  // Build type-sorted, sel-limited nlist expected by the .pt2 model
  at::Tensor firstneigh_tensor =
      buildTypeSortedNlist<double>(nlist_data.jlist, coord_d, datype, sel, nloc,
                                   mixed_types)
          .to(device);

  // Build mapping tensor.
  // NOTE: must .clone() because the local vector goes out of scope before
  // run_model is called, and torch::from_blob does not copy the data.
  at::Tensor mapping_tensor;
  if (lmp_list.mapping) {
    std::vector<std::int64_t> mapping(nall_real);
    for (int ii = 0; ii < nall_real; ii++) {
      mapping[ii] = fwd_map[lmp_list.mapping[bkw_map[ii]]];
    }
    mapping_tensor =
        torch::from_blob(mapping.data(), {1, nall_real}, int_option)
            .clone()
            .to(device);
  } else {
    // Default identity mapping for local atoms
    std::vector<std::int64_t> mapping(nall_real);
    for (int ii = 0; ii < nall_real; ii++) {
      mapping[ii] = ii;
    }
    mapping_tensor =
        torch::from_blob(mapping.data(), {1, nall_real}, int_option)
            .clone()
            .to(device);
  }

  // Build fparam/aparam tensors (cast to float64 for the model)
  auto valuetype_options = std::is_same<VALUETYPE, float>::value
                               ? torch::TensorOptions().dtype(torch::kFloat32)
                               : torch::TensorOptions().dtype(torch::kFloat64);
  at::Tensor fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<std::int64_t>(fparam.size())},
                         valuetype_options)
            .to(torch::kFloat64)
            .to(device);
  } else {
    fparam_tensor = torch::zeros({0}, options).to(device);
  }

  at::Tensor aparam_tensor;
  if (!aparam_.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam_.data()),
            {1, nloc, static_cast<std::int64_t>(aparam_.size()) / nloc},
            valuetype_options)
            .to(torch::kFloat64)
            .to(device);
  } else {
    aparam_tensor = torch::zeros({0}, options).to(device);
  }

  // Run the .pt2 model
  auto flat_outputs = run_model(coord_Tensor, atype_Tensor, firstneigh_tensor,
                                mapping_tensor, fparam_tensor, aparam_tensor);

  // Map flat outputs to internal keys
  std::map<std::string, torch::Tensor> output_map;
  extract_outputs(output_map, flat_outputs);

  // Extract energy: energy_redu (nf, 1)
  torch::Tensor flat_energy_ =
      output_map["energy_redu"].view({-1}).to(torch::kCPU);
  ener.assign(flat_energy_.data_ptr<ENERGYTYPE>(),
              flat_energy_.data_ptr<ENERGYTYPE>() + flat_energy_.numel());

  // Extract force: energy_derv_r (nf, nall, 1, 3) -> squeeze dim -2 -> (nf,
  // nall, 3)
  torch::Tensor force_tensor =
      output_map["energy_derv_r"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_force_ = force_tensor.to(torch::kCPU);
  dforce.assign(cpu_force_.data_ptr<VALUETYPE>(),
                cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());

  // Extract virial: energy_derv_c_redu (nf, 1, 9) -> squeeze dim -2 -> (nf, 9)
  torch::Tensor virial_tensor =
      output_map["energy_derv_c_redu"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = virial_tensor.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());

  // bkw map: map force from real atoms back to full atom list (including
  // NULL-type)
  force.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(force, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);

  if (atomic) {
    // Extract atom_energy: energy (nf, nloc, 1)
    torch::Tensor atom_energy_tensor =
        output_map["energy"].view({-1}).to(floatType);
    torch::Tensor cpu_atom_energy_ = atom_energy_tensor.to(torch::kCPU);
    datom_energy.resize(nall_real, 0.0);
    datom_energy.assign(
        cpu_atom_energy_.data_ptr<VALUETYPE>(),
        cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());

    // Extract atom_virial: energy_derv_c (nf, nall, 1, 9) -> squeeze dim -2 ->
    // (nf, nall, 9)
    torch::Tensor atom_virial_tensor =
        output_map["energy_derv_c"].squeeze(-2).view({-1}).to(floatType);
    torch::Tensor cpu_atom_virial_ = atom_virial_tensor.to(torch::kCPU);
    datom_virial.assign(
        cpu_atom_virial_.data_ptr<VALUETYPE>(),
        cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());

    atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
    atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
    select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
                          fwd_map.size(), nall_real);
    select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
                          fwd_map.size(), nall_real);
  }
}

template void DeepPotPTExpt::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);
template void DeepPotPTExpt::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPTExpt::compute(ENERGYVTYPE& ener,
                            std::vector<VALUETYPE>& force,
                            std::vector<VALUETYPE>& virial,
                            std::vector<VALUETYPE>& atom_energy,
                            std::vector<VALUETYPE>& atom_virial,
                            const std::vector<VALUETYPE>& coord,
                            const std::vector<int>& atype,
                            const std::vector<VALUETYPE>& box,
                            const std::vector<VALUETYPE>& fparam,
                            const std::vector<VALUETYPE>& aparam,
                            const bool atomic) {
  int natoms = atype.size();
  int nframes = coord.size() / (natoms * 3);
  if (nframes > 1) {
    // Multi-frame: loop over frames and concatenate
    compute_nframes(ener, force, virial, atom_energy, atom_virial, nframes,
                    coord, atype, box, fparam, aparam, atomic);
    return;
  }
  // The .pt2 model only contains forward_common_lower, which requires
  // nlist as input. We must build the nlist in C++ and fold back the
  // extended-region outputs to local atoms.
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }

  // Always use float64 for model inputs — the .pt2 model is compiled with
  // float64 and AOTInductor does not auto-cast.
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    floatType = torch::kFloat32;
  }
  auto int_options = torch::TensorOptions().dtype(torch::kInt64);

  // 1. Handle box: if empty (NoPbc), create a fake box large enough
  std::vector<double> coord_d(coord.begin(), coord.end());
  std::vector<double> box_d(box.begin(), box.end());
  if (box_d.empty()) {
    // Create a fake orthorhombic box that contains all atoms with margin
    double min_x = coord_d[0], max_x = coord_d[0];
    double min_y = coord_d[1], max_y = coord_d[1];
    double min_z = coord_d[2], max_z = coord_d[2];
    for (int ii = 1; ii < natoms; ++ii) {
      min_x = std::min(min_x, coord_d[ii * 3 + 0]);
      max_x = std::max(max_x, coord_d[ii * 3 + 0]);
      min_y = std::min(min_y, coord_d[ii * 3 + 1]);
      max_y = std::max(max_y, coord_d[ii * 3 + 1]);
      min_z = std::min(min_z, coord_d[ii * 3 + 2]);
      max_z = std::max(max_z, coord_d[ii * 3 + 2]);
    }
    box_d.resize(9, 0.0);
    box_d[0] = (max_x - min_x) + 2.0 * rcut;
    box_d[4] = (max_y - min_y) + 2.0 * rcut;
    box_d[8] = (max_z - min_z) + 2.0 * rcut;
  }

  // 2. Extend coords with ghosts
  std::vector<double> coord_cpy_d;
  std::vector<int> atype_cpy, mapping_vec;
  std::vector<int> ncell, ngcell;
  {
    SimulationRegion<double> region;
    region.reinitBox(&box_d[0]);
    copy_coord(coord_cpy_d, atype_cpy, mapping_vec, ncell, ngcell, coord_d,
               atype, static_cast<float>(rcut), region);
  }

  int nloc = natoms;
  int nall = coord_cpy_d.size() / 3;

  // 3. Build neighbor list on extended coords
  std::vector<std::vector<int>> nlist_raw, nlist_r_cpy;
  {
    SimulationRegion<double> region;
    region.reinitBox(&box_d[0]);
    std::vector<int> nat_stt(3, 0), ext_stt(3), ext_end(3);
    for (int dd = 0; dd < 3; ++dd) {
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    build_nlist(nlist_raw, nlist_r_cpy, coord_cpy_d, nloc, rcut, rcut, nat_stt,
                ncell, ext_stt, ext_end, region, ncell);
  }

  // 3. Build type-sorted, sel-limited nlist (uses double coords for distances)

  // 4. Convert to tensors (always float64 for .pt2 model)
  // NOTE: must .clone() because from_blob does not copy data, and the local
  // vectors would go out of scope before run_model completes.
  at::Tensor coord_Tensor =
      torch::from_blob(coord_cpy_d.data(), {1, nall, 3}, options)
          .clone()
          .to(device);
  std::vector<std::int64_t> atype_64(atype_cpy.begin(), atype_cpy.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall}, int_options)
          .clone()
          .to(device);
  at::Tensor nlist_tensor =
      buildTypeSortedNlist<double>(nlist_raw, coord_cpy_d, atype_cpy, sel, nloc,
                                   mixed_types)
          .to(device);
  std::vector<std::int64_t> mapping_64(mapping_vec.begin(), mapping_vec.end());
  at::Tensor mapping_tensor =
      torch::from_blob(mapping_64.data(), {1, nall}, int_options)
          .clone()
          .to(device);

  // Build fparam/aparam tensors (cast to float64 for the model)
  auto valuetype_options = std::is_same<VALUETYPE, float>::value
                               ? torch::TensorOptions().dtype(torch::kFloat32)
                               : torch::TensorOptions().dtype(torch::kFloat64);
  at::Tensor fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<std::int64_t>(fparam.size())},
                         valuetype_options)
            .to(torch::kFloat64)
            .to(device);
  } else {
    fparam_tensor = torch::zeros({0}, options).to(device);
  }

  at::Tensor aparam_tensor;
  if (!aparam.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam.data()),
            {1, natoms, static_cast<std::int64_t>(aparam.size()) / natoms},
            valuetype_options)
            .to(torch::kFloat64)
            .to(device);
  } else {
    aparam_tensor = torch::zeros({0}, options).to(device);
  }

  // 5. Run the .pt2 model
  auto flat_outputs = run_model(coord_Tensor, atype_Tensor, nlist_tensor,
                                mapping_tensor, fparam_tensor, aparam_tensor);

  // 6. Map flat outputs to internal keys
  std::map<std::string, torch::Tensor> output_map;
  extract_outputs(output_map, flat_outputs);

  // 7. Extract energy
  torch::Tensor flat_energy_ =
      output_map["energy_redu"].view({-1}).to(torch::kCPU);
  ener.assign(flat_energy_.data_ptr<ENERGYTYPE>(),
              flat_energy_.data_ptr<ENERGYTYPE>() + flat_energy_.numel());

  // 8. Extract virial: energy_derv_c_redu (nf, 1, 9) -> (nf, 9)
  torch::Tensor virial_tensor =
      output_map["energy_derv_c_redu"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = virial_tensor.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());

  // 9. Extract force and fold back: energy_derv_r (nf, nall, 1, 3) -> (nf,
  // nall, 3)
  torch::Tensor force_ext =
      output_map["energy_derv_r"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_force_ext = force_ext.to(torch::kCPU);
  std::vector<VALUETYPE> extended_force(
      cpu_force_ext.data_ptr<VALUETYPE>(),
      cpu_force_ext.data_ptr<VALUETYPE>() + cpu_force_ext.numel());
  fold_back(force, extended_force, mapping_vec, nloc, nall, 3, nframes);

  if (atomic) {
    // atom_energy: energy (nf, nloc, 1) — already on local atoms
    torch::Tensor atom_energy_tensor =
        output_map["energy"].view({-1}).to(floatType);
    torch::Tensor cpu_atom_energy_ = atom_energy_tensor.to(torch::kCPU);
    atom_energy.assign(
        cpu_atom_energy_.data_ptr<VALUETYPE>(),
        cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());

    // atom_virial: energy_derv_c (nf, nall, 1, 9) -> (nf, nall, 9)
    // fold back to local atoms
    torch::Tensor atom_virial_ext =
        output_map["energy_derv_c"].squeeze(-2).view({-1}).to(floatType);
    torch::Tensor cpu_atom_virial_ext = atom_virial_ext.to(torch::kCPU);
    std::vector<VALUETYPE> extended_atom_virial(
        cpu_atom_virial_ext.data_ptr<VALUETYPE>(),
        cpu_atom_virial_ext.data_ptr<VALUETYPE>() +
            cpu_atom_virial_ext.numel());
    fold_back(atom_virial, extended_atom_virial, mapping_vec, nloc, nall, 9,
              nframes);
  }
}

template void DeepPotPTExpt::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);
template void DeepPotPTExpt::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPTExpt::compute_nframes(ENERGYVTYPE& ener,
                                    std::vector<VALUETYPE>& force,
                                    std::vector<VALUETYPE>& virial,
                                    std::vector<VALUETYPE>& atom_energy,
                                    std::vector<VALUETYPE>& atom_virial,
                                    const int nframes,
                                    const std::vector<VALUETYPE>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<VALUETYPE>& box,
                                    const std::vector<VALUETYPE>& fparam,
                                    const std::vector<VALUETYPE>& aparam,
                                    const bool atomic) {
  int natoms = atype.size();
  int dap = aparam.empty() ? 0 : static_cast<int>(aparam.size()) / nframes;
  int dfp = fparam.empty() ? 0 : static_cast<int>(fparam.size()) / nframes;
  ener.clear();
  force.clear();
  virial.clear();
  if (atomic) {
    atom_energy.clear();
    atom_virial.clear();
  }
  for (int ff = 0; ff < nframes; ++ff) {
    size_t s_ff = static_cast<size_t>(ff);
    size_t s_natoms = static_cast<size_t>(natoms);
    std::vector<VALUETYPE> frame_coord(
        coord.begin() + s_ff * s_natoms * 3,
        coord.begin() + (s_ff + 1) * s_natoms * 3);
    std::vector<VALUETYPE> frame_box;
    if (!box.empty()) {
      frame_box.assign(box.begin() + s_ff * 9, box.begin() + (s_ff + 1) * 9);
    }
    std::vector<VALUETYPE> frame_fparam;
    if (!fparam.empty()) {
      size_t s_dfp = static_cast<size_t>(dfp);
      frame_fparam.assign(fparam.begin() + s_ff * s_dfp,
                          fparam.begin() + (s_ff + 1) * s_dfp);
    }
    std::vector<VALUETYPE> frame_aparam;
    if (!aparam.empty()) {
      size_t s_dap = static_cast<size_t>(dap);
      frame_aparam.assign(aparam.begin() + s_ff * s_dap,
                          aparam.begin() + (s_ff + 1) * s_dap);
    }
    std::vector<ENERGYTYPE> frame_ener;
    std::vector<VALUETYPE> frame_force, frame_virial, frame_ae, frame_av;
    compute(frame_ener, frame_force, frame_virial, frame_ae, frame_av,
            frame_coord, atype, frame_box, frame_fparam, frame_aparam, atomic);
    ener.insert(ener.end(), frame_ener.begin(), frame_ener.end());
    force.insert(force.end(), frame_force.begin(), frame_force.end());
    virial.insert(virial.end(), frame_virial.begin(), frame_virial.end());
    if (atomic) {
      atom_energy.insert(atom_energy.end(), frame_ae.begin(), frame_ae.end());
      atom_virial.insert(atom_virial.end(), frame_av.begin(), frame_av.end());
    }
  }
}

void DeepPotPTExpt::get_type_map(std::string& type_map_str) {
  for (const auto& t : type_map) {
    type_map_str += t;
    type_map_str += " ";
  }
}

// forward to template method
void DeepPotPTExpt::computew(std::vector<double>& ener,
                             std::vector<double>& force,
                             std::vector<double>& virial,
                             std::vector<double>& atom_energy,
                             std::vector<double>& atom_virial,
                             const std::vector<double>& coord,
                             const std::vector<int>& atype,
                             const std::vector<double>& box,
                             const std::vector<double>& fparam,
                             const std::vector<double>& aparam,
                             const bool atomic) {
  translate_error([&] {
    compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
            fparam, aparam, atomic);
  });
}
void DeepPotPTExpt::computew(std::vector<double>& ener,
                             std::vector<float>& force,
                             std::vector<float>& virial,
                             std::vector<float>& atom_energy,
                             std::vector<float>& atom_virial,
                             const std::vector<float>& coord,
                             const std::vector<int>& atype,
                             const std::vector<float>& box,
                             const std::vector<float>& fparam,
                             const std::vector<float>& aparam,
                             const bool atomic) {
  translate_error([&] {
    compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
            fparam, aparam, atomic);
  });
}
void DeepPotPTExpt::computew(std::vector<double>& ener,
                             std::vector<double>& force,
                             std::vector<double>& virial,
                             std::vector<double>& atom_energy,
                             std::vector<double>& atom_virial,
                             const std::vector<double>& coord,
                             const std::vector<int>& atype,
                             const std::vector<double>& box,
                             const int nghost,
                             const InputNlist& inlist,
                             const int& ago,
                             const std::vector<double>& fparam,
                             const std::vector<double>& aparam,
                             const bool atomic) {
  translate_error([&] {
    compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
            nghost, inlist, ago, fparam, aparam, atomic);
  });
}
void DeepPotPTExpt::computew(std::vector<double>& ener,
                             std::vector<float>& force,
                             std::vector<float>& virial,
                             std::vector<float>& atom_energy,
                             std::vector<float>& atom_virial,
                             const std::vector<float>& coord,
                             const std::vector<int>& atype,
                             const std::vector<float>& box,
                             const int nghost,
                             const InputNlist& inlist,
                             const int& ago,
                             const std::vector<float>& fparam,
                             const std::vector<float>& aparam,
                             const bool atomic) {
  translate_error([&] {
    compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
            nghost, inlist, ago, fparam, aparam, atomic);
  });
}
template <typename VALUETYPE>
void DeepPotPTExpt::compute_mixed_type_impl(
    std::vector<double>& ener,
    std::vector<VALUETYPE>& force,
    std::vector<VALUETYPE>& virial,
    std::vector<VALUETYPE>& atom_energy,
    std::vector<VALUETYPE>& atom_virial,
    const int& nframes,
    const std::vector<VALUETYPE>& coord,
    const std::vector<int>& atype,
    const std::vector<VALUETYPE>& box,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam,
    const bool atomic) {
  // Mixed-type: atype has nframes * natoms elements.
  // Loop over frames, each with its own atype slice.
  int natoms = static_cast<int>(atype.size()) / nframes;
  int dap = aparam.empty() ? 0 : static_cast<int>(aparam.size()) / nframes;
  int dfp = fparam.empty() ? 0 : static_cast<int>(fparam.size()) / nframes;
  ener.clear();
  force.clear();
  virial.clear();
  if (atomic) {
    atom_energy.clear();
    atom_virial.clear();
  }
  for (int ff = 0; ff < nframes; ++ff) {
    size_t s_ff = static_cast<size_t>(ff);
    size_t s_natoms = static_cast<size_t>(natoms);
    std::vector<VALUETYPE> frame_coord(
        coord.begin() + s_ff * s_natoms * 3,
        coord.begin() + (s_ff + 1) * s_natoms * 3);
    std::vector<int> frame_atype(atype.begin() + s_ff * s_natoms,
                                 atype.begin() + (s_ff + 1) * s_natoms);
    std::vector<VALUETYPE> frame_box;
    if (!box.empty()) {
      frame_box.assign(box.begin() + s_ff * 9, box.begin() + (s_ff + 1) * 9);
    }
    std::vector<VALUETYPE> frame_fparam;
    if (!fparam.empty()) {
      size_t s_dfp = static_cast<size_t>(dfp);
      frame_fparam.assign(fparam.begin() + s_ff * s_dfp,
                          fparam.begin() + (s_ff + 1) * s_dfp);
    }
    std::vector<VALUETYPE> frame_aparam;
    if (!aparam.empty()) {
      size_t s_dap = static_cast<size_t>(dap);
      frame_aparam.assign(aparam.begin() + s_ff * s_dap,
                          aparam.begin() + (s_ff + 1) * s_dap);
    }
    std::vector<ENERGYTYPE> frame_ener;
    std::vector<VALUETYPE> frame_force, frame_virial, frame_ae, frame_av;
    compute(frame_ener, frame_force, frame_virial, frame_ae, frame_av,
            frame_coord, frame_atype, frame_box, frame_fparam, frame_aparam,
            atomic);
    ener.insert(ener.end(), frame_ener.begin(), frame_ener.end());
    force.insert(force.end(), frame_force.begin(), frame_force.end());
    virial.insert(virial.end(), frame_virial.begin(), frame_virial.end());
    if (atomic) {
      atom_energy.insert(atom_energy.end(), frame_ae.begin(), frame_ae.end());
      atom_virial.insert(atom_virial.end(), frame_av.begin(), frame_av.end());
    }
  }
}

void DeepPotPTExpt::computew_mixed_type(std::vector<double>& ener,
                                        std::vector<double>& force,
                                        std::vector<double>& virial,
                                        std::vector<double>& atom_energy,
                                        std::vector<double>& atom_virial,
                                        const int& nframes,
                                        const std::vector<double>& coord,
                                        const std::vector<int>& atype,
                                        const std::vector<double>& box,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam,
                                        const bool atomic) {
  translate_error([&] {
    compute_mixed_type_impl(ener, force, virial, atom_energy, atom_virial,
                            nframes, coord, atype, box, fparam, aparam, atomic);
  });
}
void DeepPotPTExpt::computew_mixed_type(std::vector<double>& ener,
                                        std::vector<float>& force,
                                        std::vector<float>& virial,
                                        std::vector<float>& atom_energy,
                                        std::vector<float>& atom_virial,
                                        const int& nframes,
                                        const std::vector<float>& coord,
                                        const std::vector<int>& atype,
                                        const std::vector<float>& box,
                                        const std::vector<float>& fparam,
                                        const std::vector<float>& aparam,
                                        const bool atomic) {
  translate_error([&] {
    compute_mixed_type_impl(ener, force, virial, atom_energy, atom_virial,
                            nframes, coord, atype, box, fparam, aparam, atomic);
  });
}
#endif
