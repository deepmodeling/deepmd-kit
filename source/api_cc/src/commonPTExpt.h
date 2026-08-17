// SPDX-License-Identifier: LGPL-3.0-or-later
// Shared utilities for pt_expt (.pt2 / AOTInductor) backend classes.
// Provides: JSON parser, ZIP archive reader, type-sorted nlist builder,
// and helpers for the with-comm dual-artifact layout.
#pragma once

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"  // for remap_comm_sendlist
#include "errors.h"
#include "neighbor_list.h"

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

inline std::vector<double> read_default_chg_spin(const JsonValue& metadata,
                                                 const int dim_chg_spin) {
  std::vector<double> default_chg_spin;
  if (dim_chg_spin <= 0) {
    return default_chg_spin;
  }
  if (!metadata.obj_val.count("default_chg_spin")) {
    throw deepmd::deepmd_exception(
        "Model requires charge/spin conditions but default_chg_spin is "
        "missing from metadata.");
  }
  for (const auto& v : metadata["default_chg_spin"].as_array()) {
    default_chg_spin.push_back(v.as_double());
  }
  if (static_cast<int>(default_chg_spin.size()) != dim_chg_spin) {
    throw deepmd::deepmd_exception("default_chg_spin length (" +
                                   std::to_string(default_chg_spin.size()) +
                                   ") does not match dim_chg_spin (" +
                                   std::to_string(dim_chg_spin) + ").");
  }
  return default_chg_spin;
}

/**
 * @brief Read the row range each value of a charge state indexes.
 *
 * A charge-conditioned descriptor embeds the condition by gathering one row
 * of each embedding table, so the archive records the half-open range of each
 * table. An archive frozen before the ranges were recorded names none, and
 * the boundary then checks only the width, as it did before.
 *
 * @param[in] metadata Parsed archive metadata.
 * @return One ``{low, high}`` pair per value, empty when the archive names
 *   no ranges.
 **/
inline std::vector<std::pair<double, double>> read_chg_spin_table_ranges(
    const JsonValue& metadata) {
  std::vector<std::pair<double, double>> ranges;
  if (!metadata.obj_val.count("chg_spin_table_ranges") ||
      metadata["chg_spin_table_ranges"].type == JsonValue::Null) {
    return ranges;
  }
  for (const auto& bounds : metadata["chg_spin_table_ranges"].as_array()) {
    const auto& pair = bounds.as_array();
    if (pair.size() != 2) {
      throw deepmd::deepmd_exception(
          "chg_spin_table_ranges must hold a [low, high) pair per value.");
    }
    ranges.emplace_back(pair[0].as_double(), pair[1].as_double());
  }
  return ranges;
}

/**
 * @brief Reject a charge state that addresses no row of the model's tables.
 *
 * The gather is not bounds-checked and the index is truncated from a double,
 * so a fractional value would silently land on a neighbouring row and an
 * out-of-range value would read past the table. An archive that names no
 * ranges is left unchecked.
 *
 * @param[in] charge_spin One or more states laid out end to end.
 * @param[in] ranges Half-open row range of each value of one state.
 **/
inline void check_charge_spin_domain(
    const std::vector<double>& charge_spin,
    const std::vector<std::pair<double, double>>& ranges) {
  if (ranges.empty() || charge_spin.empty()) {
    return;
  }
  const std::size_t width = ranges.size();
  static const char* kFields[] = {"charge", "multiplicity"};
  for (std::size_t ii = 0; ii < charge_spin.size(); ++ii) {
    const std::size_t column = ii % width;
    const double value = charge_spin[ii];
    const std::string field =
        column < 2 ? kFields[column]
                   : "charge_spin value " + std::to_string(column);
    if (!std::isfinite(value) || value != std::floor(value)) {
      throw deepmd::deepmd_exception("the " + field +
                                     " indexes an embedding table row and "
                                     "must be an integer, got " +
                                     std::to_string(value));
    }
    if (value < ranges[column].first || value >= ranges[column].second) {
      throw deepmd::deepmd_exception(
          "the " + field + " must lie in [" +
          std::to_string(static_cast<long>(ranges[column].first)) + ", " +
          std::to_string(static_cast<long>(ranges[column].second)) + "), got " +
          std::to_string(static_cast<long>(value)));
    }
  }
}

/**
 * @brief Validate a charge/spin condition supplied with an inference call.
 *
 * The condition holds either one frame's values, broadcast to every frame, or
 * one set per frame. Whether a call may name a condition of its own depends
 * on how the model receives it. A lower that reads the condition as an
 * ordinary input takes a different one on every call. A model whose condition
 * instead lives in the constants of its compiled tables, and a backend that
 * marshals only the condition in force, both serve one state at a time; there
 * a supplied condition can only restate the state in force, and anything else
 * is rejected rather than silently ignored.
 *
 * @param[in] charge_spin The condition supplied by the caller. Empty selects
 *   the state in force and is always accepted.
 * @param[in] nframes Number of frames the call evaluates.
 * @param[in] settable_chg_spin Width of a charge state the model accepts.
 * @param[in] applied_per_call Whether the call marshals the condition into
 *   the forward pass instead of serving the state in force.
 * @param[in] installed The state in force, of width ``settable_chg_spin``.
 * @param[in] ranges Half-open row range of each value, from the archive.
 **/
inline void check_call_charge_spin(
    const std::vector<double>& charge_spin,
    const int nframes,
    const int settable_chg_spin,
    const bool applied_per_call,
    const std::vector<double>& installed,
    const std::vector<std::pair<double, double>>& ranges =
        std::vector<std::pair<double, double>>()) {
  if (charge_spin.empty()) {
    return;
  }
  check_charge_spin_domain(charge_spin, ranges);
  const std::size_t width = static_cast<std::size_t>(settable_chg_spin);
  if (charge_spin.size() != width &&
      charge_spin.size() != width * static_cast<std::size_t>(nframes)) {
    throw deepmd::deepmd_exception(
        "charge_spin has " + std::to_string(charge_spin.size()) +
        " values but the model expects dim_chg_spin=" +
        std::to_string(settable_chg_spin) + " (per frame) or " +
        std::to_string(settable_chg_spin * nframes) + " (for " +
        std::to_string(nframes) + " frames).");
  }
  if (applied_per_call) {
    return;
  }
  if (installed.size() != width) {
    throw deepmd::deepmd_exception(
        "the model serves one charge/spin state at a time but holds none of "
        "the " +
        std::to_string(settable_chg_spin) + " values it accepts.");
  }
  // Charge and multiplicity are integer-valued categorical indices carried as
  // double, so a condition either is the state in force or is not.
  for (std::size_t ii = 0; ii < charge_spin.size(); ++ii) {
    if (charge_spin[ii] != installed[ii % width]) {
      throw deepmd::deepmd_exception(
          "the charge/spin condition supplied with this call differs from "
          "the state the model serves. This model serves one state at a "
          "time, held in the constants of its compiled tables or fixed for "
          "the whole run, so the state must be chosen with set_charge_spin "
          "before inference rather than named per call.");
    }
  }
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
// With-comm artifact extraction
//
// GNN .pt2 archives carry a nested ``extra/forward_lower_with_comm.pt2``
// alongside the regular forward_lower artifact.  AOTInductor's
// ``ModelPackageLoader`` reads .pt2 files from disk, so to load the
// nested artifact we extract it to a temp file.
// ============================================================================

/**
 * @brief RAII handle for a temp file on disk.
 *
 * Used to hold the extracted with-comm .pt2 artifact for the lifetime
 * of the loader.  Destructor unlinks the file.
 */
class TempFile {
 public:
  TempFile() = default;
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  TempFile(TempFile&& other) noexcept : path_(std::move(other.path_)) {
    other.path_.clear();
  }
  TempFile& operator=(TempFile&& other) noexcept {
    if (this != &other) {
      cleanup();
      path_ = std::move(other.path_);
      other.path_.clear();
    }
    return *this;
  }
  ~TempFile() { cleanup(); }

  const std::string& path() const { return path_; }
  bool empty() const { return path_.empty(); }

  /**
   * @brief Write the content of an existing .pt2 ZIP entry to a temp
   * file and return a TempFile owning that path.
   *
   * The temp file is created via ``mkstemp(3)`` (atomic, unique,
   * 0600 permissions) under the system tempdir (TMPDIR or /tmp), and is
   * named after the entry it holds so that a file left behind by a crash
   * says which artifact it came from.
   */
  static TempFile from_zip_entry(const std::string& outer_pt2_path,
                                 const std::string& entry_name) {
    std::string content = read_zip_entry(outer_pt2_path, entry_name);
    const char* tmpdir = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmpdir ? tmpdir : "/tmp") + "/dp_pt2_" +
                       entry_stem(entry_name) + "_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = mkstemp(buf.data());
    if (fd < 0) {
      throw deepmd::deepmd_exception(
          "Failed to create temp file for nested .pt2 artifact: " + tmpl);
    }
    std::string path(buf.data());
    // Write content to the fd so we don't race with another process
    // opening the same path.
    ssize_t written = 0;
    const char* p = content.data();
    ssize_t remain = static_cast<ssize_t>(content.size());
    while (remain > 0) {
      ssize_t n = ::write(fd, p + written, static_cast<size_t>(remain));
      if (n < 0) {
        ::close(fd);
        ::unlink(path.c_str());
        throw deepmd::deepmd_exception(
            "Failed to write nested .pt2 artifact to temp file: " + path);
      }
      written += n;
      remain -= n;
    }
    ::close(fd);
    TempFile tf;
    tf.path_ = std::move(path);
    return tf;
  }

 private:
  /**
   * @brief The base name of a ZIP entry, without directories or extension
   * and reduced to characters a file name carries safely.
   */
  static std::string entry_stem(const std::string& entry_name) {
    const std::size_t slash = entry_name.find_last_of('/');
    std::string stem =
        slash == std::string::npos ? entry_name : entry_name.substr(slash + 1);
    const std::size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos) {
      stem.erase(dot);
    }
    for (char& c : stem) {
      if (!std::isalnum(static_cast<unsigned char>(c))) {
        c = '_';
      }
    }
    return stem;
  }

  void cleanup() {
    if (!path_.empty()) {
      ::unlink(path_.c_str());
      path_.clear();
    }
  }
  std::string path_;
};

// ============================================================================
// Communication tensor packing for the with-comm artifact
//
// The with-comm AOTInductor artifact accepts comm tensors as 8 additional
// positional inputs (after the regular 4-6 inputs) in this canonical order:
//   send_list (nswap, int64 ptr-array packed as int64 tensor)
//   send_proc (nswap, int32)
//   recv_proc (nswap, int32)
//   send_num  (nswap, int32)
//   recv_num  (nswap, int32)
//   communicator (1, int64 — MPI handle as opaque int)
//   nlocal    (scalar int32)
//   nghost    (scalar int32)
// This mirrors deepmd_export::border_op's argument order in
// deepmd/pt_expt/utils/comm.py.
// ============================================================================

/**
 * @brief Build the 8 comm-tensor positional inputs from LAMMPS data.
 */
inline std::vector<at::Tensor> build_comm_tensors_positional(
    const InputNlist& lmp_list,
    int** sendlist,
    int* sendnum,
    int* recvnum,
    int nlocal,
    int nghost) {
  int nswap = lmp_list.nswap;
  auto int32_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
  auto int64_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

  // The with-comm AOTInductor artifact is compiled assuming 16-byte-aligned
  // inputs (the freeze-time sample comm tensors are torch-allocated). LAMMPS'
  // raw send/recv arrays and the MPI handle carry only their natural element
  // alignment, so wrapping them with ``from_blob`` would force AOTInductor to
  // copy each input to an aligned buffer on every step (a per-step warning and
  // copy). ``clone`` materialises them in torch-allocated aligned storage; the
  // pointer values inside ``sendlist`` are copied verbatim and still address
  // the live LAMMPS swap buffers. The clones are tiny (``nswap`` elements), so
  // the one-time copy is negligible.
  at::Tensor sendlist_tensor =
      torch::from_blob(static_cast<void*>(sendlist), {nswap}, int64_option)
          .clone();
  at::Tensor sendproc_tensor =
      torch::from_blob(lmp_list.sendproc, {nswap}, int32_option).clone();
  at::Tensor recvproc_tensor =
      torch::from_blob(lmp_list.recvproc, {nswap}, int32_option).clone();
  at::Tensor sendnum_tensor =
      torch::from_blob(sendnum, {nswap}, int32_option).clone();
  at::Tensor recvnum_tensor =
      torch::from_blob(recvnum, {nswap}, int32_option).clone();

  std::int64_t null_communicator = 0;
  at::Tensor communicator_tensor;
  if (lmp_list.world == nullptr) {
    communicator_tensor =
        torch::from_blob(&null_communicator, {1}, int64_option).clone();
  } else {
    communicator_tensor =
        torch::from_blob(const_cast<void*>(lmp_list.world), {1}, int64_option)
            .clone();
  }

  at::Tensor nlocal_tensor = torch::tensor(nlocal, int32_option);
  at::Tensor nghost_tensor = torch::tensor(nghost, int32_option);

  return {sendlist_tensor, sendproc_tensor,     recvproc_tensor, sendnum_tensor,
          recvnum_tensor,  communicator_tensor, nlocal_tensor,   nghost_tensor};
}

/**
 * @brief Build the 8 comm-tensor positional inputs with NULL-type-atom
 * remapping.  When ``select_real_atoms_coord`` filters atoms (atype <
 * 0), ``fwd_map`` translates original sendlist indices into real-atom
 * indices (with ``-1`` for filtered).  Mirrors
 * ``commonPT.h::build_comm_dict_with_virtual_atoms``.  The remapped
 * storage must outlive the returned tensors.
 */
inline std::vector<at::Tensor> build_comm_tensors_positional_with_virtual_atoms(
    const InputNlist& lmp_list,
    const std::vector<int>& fwd_map,
    int nlocal,
    int nghost,
    std::vector<std::vector<int>>& remapped_sendlist,
    std::vector<int*>& remapped_sendlist_ptrs,
    std::vector<int>& remapped_sendnum,
    std::vector<int>& remapped_recvnum) {
  remap_comm_sendlist(remapped_sendlist, remapped_sendnum, remapped_recvnum,
                      lmp_list, fwd_map);
  int nswap = lmp_list.nswap;
  remapped_sendlist_ptrs.resize(nswap);
  for (int s = 0; s < nswap; ++s) {
    remapped_sendlist_ptrs[s] = remapped_sendlist[s].data();
  }
  return build_comm_tensors_positional(lmp_list, remapped_sendlist_ptrs.data(),
                                       remapped_sendnum.data(),
                                       remapped_recvnum.data(), nlocal, nghost);
}

// ============================================================================
// Charge-state fold — the runtime charge/spin condition of a compressed model.
// ============================================================================

/**
 * @brief The frozen tables through which a compressed descriptor carries its
 * charge/spin condition, as a rebuild that can be re-run at any time.
 *
 * A compressed charge-conditioned descriptor evaluates its frame condition
 * once, when the model is frozen, into a handful of tables. Those tables
 * reach a compiled lower as module constants, so serving a different
 * condition means rebuilding them and writing them over those constants
 * rather than re-evaluating the condition on every step. The archive
 * therefore ships a second compiled artifact that performs the rebuild,
 * together with the name of the constant each of its outputs replaces.
 *
 * Every lower lifts its constants independently, so the names hold only for
 * the lower they were resolved against at freeze time. Only a compressed
 * DPA4C descriptor folds a charge state, and that family never carries
 * message passing across ranks, so an archive with a fold holds exactly one
 * lower and the question of a second set of names does not arise.
 *
 * An archive without the rebuild leaves the fold inactive. That is the
 * ordinary case: an uncompressed model reads its condition as a plain input,
 * so nothing needs rebuilding.
 */
class ChargeStateFold {
 public:
  /**
   * @brief Load the rebuild an archive declares, if it declares one.
   *
   * The constant-name field is the archive's claim that the rebuild ships
   * with it, so an archive that declares the names and cannot supply the
   * rebuild is malformed and fails here rather than degrading silently.
   *
   * @param[in] model_path Path to the .pt2 archive.
   * @param[in] metadata Parsed archive metadata.
   * @param[in] gpu_enabled Whether the lower was loaded on a GPU.
   * @param[in] gpu_id The GPU the lower was loaded on.
   * @return The fold, or ``nullptr`` when the archive declares none.
   **/
  static std::unique_ptr<ChargeStateFold> load(const std::string& model_path,
                                               const JsonValue& metadata,
                                               const bool gpu_enabled,
                                               const int gpu_id) {
    if (!metadata.obj_val.count("charge_state_constants")) {
      return nullptr;
    }
    if (metadata.obj_val.count("has_comm_artifact") &&
        metadata["has_comm_artifact"].as_bool()) {
      throw deepmd::deepmd_exception(
          "the archive ships a charge-state fold beside a with-comm lower; "
          "the fold names the constants of one lower only, so the second "
          "would keep serving the condition it was frozen against");
    }
    std::unique_ptr<ChargeStateFold> fold(new ChargeStateFold());
    fold->constants_ = read_names(metadata, "charge_state_constants");
    fold->tempfile_ = std::make_unique<TempFile>(
        TempFile::from_zip_entry(model_path, "extra/charge_state.pt2"));
    fold->loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
        fold->tempfile_->path(), "model", false, 1,
        gpu_enabled ? static_cast<c10::DeviceIndex>(gpu_id)
                    : static_cast<c10::DeviceIndex>(-1));
    return fold;
  }

  /**
   * @brief Rebuild the tables for a condition and write them over the
   * constants of the lower.
   *
   * @param[in] charge_spin The condition.
   * @param[in] device The device the lower was loaded on.
   * @param[in,out] target The lower whose constants carry the condition.
   **/
  void apply(const std::vector<double>& charge_spin,
             const torch::Device& device,
             torch::inductor::AOTIModelPackageLoader& target) const {
    // The rebuild consumes the condition in the (1, dim) float32 layout the
    // inference lower would receive.
    std::vector<float> state(charge_spin.begin(), charge_spin.end());
    torch::Tensor state_tensor =
        torch::from_blob(state.data(),
                         {1, static_cast<std::int64_t>(state.size())},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .clone()
            .to(device);
    std::vector<torch::Tensor> tables = loader_->run({state_tensor});
    if (tables.size() != constants_.size()) {
      throw deepmd::deepmd_exception(
          "the charge-state rebuild returned " + std::to_string(tables.size()) +
          " tables but the archive names " + std::to_string(constants_.size()) +
          " constants; it cannot serve a runtime charge state");
    }
    // The inactive buffer is a complete model image. A partial inactive update
    // copies tensor constants and buffers but omits ordinary parameters, so it
    // cannot be swapped into service safely.
    auto* runner = target.get_runner();
    auto constants = runner->extract_constants_map(/*use_inactive=*/false);
    for (size_t ii = 0; ii < tables.size(); ++ii) {
      // An unnamed output belongs to a mechanism this model has disabled and
      // has no constant to reach.
      if (!constants_[ii].empty()) {
        constants[constants_[ii]] = tables[ii];
      }
    }
    target.load_constants(constants, /*use_inactive=*/true,
                          /*check_full_update=*/true);
    runner->swap_constant_buffer();
    runner->free_inactive_constant_buffer();
  }

 private:
  ChargeStateFold() = default;

  static std::vector<std::string> read_names(const JsonValue& metadata,
                                             const std::string& key) {
    std::vector<std::string> names;
    for (const auto& v : metadata[key].as_array()) {
      names.push_back(v.as_string());
    }
    return names;
  }

  std::vector<std::string> constants_;
  std::unique_ptr<TempFile> tempfile_;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
};

}  // namespace ptexpt
}  // namespace deepmd
