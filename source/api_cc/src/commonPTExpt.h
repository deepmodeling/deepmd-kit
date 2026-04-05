// SPDX-License-Identifier: LGPL-3.0-or-later
// Shared utilities for pt_expt (.pt2 / AOTInductor) backend classes.
// Provides: JSON parser, ZIP archive reader, and type-sorted nlist builder.
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "errors.h"

namespace deepmd {
namespace ptexpt {

// ============================================================================
// Minimal JSON value parser for reading metadata from .pt2 archives.
// Supports: strings, numbers, booleans, arrays, objects.
// ============================================================================

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

inline JsonValue parse_json(const std::string& s) {
  JsonParser parser(s);
  return parser.parse();
}

// ============================================================================
// ZIP archive reader — reads a file from a ZIP archive.
// ============================================================================

inline std::string read_zip_entry(const std::string& zip_path,
                                  const std::string& entry_name) {
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

  // Handle ZIP64
  if (cd_offset == 0xFFFFFFFF || num_entries == 0xFFFF) {
    if (eocd_pos < 20) {
      throw deepmd::deepmd_exception(
          "Invalid ZIP64 file (truncated EOCD locator): " + zip_path);
    }
    size_t zip64_locator_pos = eocd_pos - 20;
    if (content[zip64_locator_pos] == 0x50 &&
        content[zip64_locator_pos + 1] == 0x4b &&
        content[zip64_locator_pos + 2] == 0x06 &&
        content[zip64_locator_pos + 3] == 0x07) {
      uint64_t zip64_eocd_offset = 0;
      for (int b = 0; b < 8; ++b) {
        zip64_eocd_offset |= static_cast<uint64_t>(static_cast<unsigned char>(
                                 content[zip64_locator_pos + 8 + b]))
                             << (8 * b);
      }
      size_t z64_pos = static_cast<size_t>(zip64_eocd_offset);
      if (z64_pos + 56 > content.size()) {
        throw deepmd::deepmd_exception(
            "Invalid ZIP64 file (truncated EOCD record): " + zip_path);
      }
      num_entries = 0;
      for (int b = 0; b < 8; ++b) {
        num_entries |= static_cast<uint64_t>(static_cast<unsigned char>(
                           content[z64_pos + 32 + b]))
                       << (8 * b);
      }
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
    if (pos + 46 > content.size()) {
      break;
    }
    uint16_t name_len = read_u16(pos + 28);
    uint16_t extra_len = read_u16(pos + 30);
    uint16_t comment_len = read_u16(pos + 32);
    uint32_t compressed_size_u32 = read_u32(pos + 20);
    uint32_t uncompressed_size_u32 = read_u32(pos + 24);
    uint32_t local_header_offset_u32 = read_u32(pos + 42);

    uint64_t compressed_size = compressed_size_u32;
    uint64_t uncompressed_size = uncompressed_size_u32;
    uint64_t local_header_offset = local_header_offset_u32;

    std::string name = content.substr(pos + 46, name_len);

    // Handle ZIP64 extra field
    if (uncompressed_size_u32 == 0xFFFFFFFF ||
        local_header_offset_u32 == 0xFFFFFFFF) {
      size_t extra_pos = pos + 46 + name_len;
      size_t extra_end = extra_pos + extra_len;
      while (extra_pos + 4 <= extra_end) {
        uint16_t field_id = read_u16(extra_pos);
        uint16_t field_size = read_u16(extra_pos + 2);
        if (field_id == 0x0001) {
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

    // Match exact name or suffix
    bool match = (name == entry_name);
    if (!match && name.size() > entry_name.size()) {
      size_t suffix_start = name.size() - entry_name.size();
      if (name[suffix_start - 1] == '/' &&
          name.substr(suffix_start) == entry_name) {
        match = true;
      }
    }
    if (match) {
      uint16_t local_name_len = read_u16(local_header_offset + 26);
      uint16_t local_extra_len = read_u16(local_header_offset + 28);
      size_t data_offset =
          local_header_offset + 30 + local_name_len + local_extra_len;
      // PyTorch archives (.pth, .pte, .pt2) always use ZIP STORED (compression
      // method 0) for every entry. PyTorch needs to mmap tensor data directly
      // from the archive without decompression, so its C++ writer
      // (caffe2::serialize::PyTorchStreamWriter) and torch.export.save both
      // write uncompressed entries with 64-byte alignment. No decompression is
      // needed.
      return content.substr(data_offset, uncompressed_size);
    }

    pos += 46 + name_len + extra_len + comment_len;
  }

  throw deepmd::deepmd_exception("Entry not found in ZIP: " + entry_name +
                                 " in " + zip_path);
}

// ============================================================================
// Build type-sorted, sel-limited neighbor list tensor.
// ============================================================================

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
 */
template <typename VALUETYPE>
inline torch::Tensor buildTypeSortedNlist(
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
      std::vector<std::pair<VALUETYPE, int>> all_neighbors;
      for (int jj : neighbors) {
        if (jj < 0) {
          continue;
        }
        int jtype = atype_ext[jj];
        if (jtype < 0) {
          continue;
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
      std::vector<std::vector<std::pair<VALUETYPE, int>>> by_type(ntypes);
      for (int jj : neighbors) {
        if (jj < 0) {
          continue;
        }
        int jtype = atype_ext[jj];
        if (jtype < 0 || jtype >= ntypes) {
          continue;
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

}  // namespace ptexpt
}  // namespace deepmd
