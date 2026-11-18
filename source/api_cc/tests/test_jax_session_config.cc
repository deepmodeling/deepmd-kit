// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "common.h"

namespace {

using Bytes = std::vector<std::uint8_t>;

// gpu_options: per_process_gpu_memory_fraction = 0.9, allow_growth = true;
// followed by allow_soft_placement = true.
const Bytes kTail = {0x32, 0x0b, 0x09, 0xcd, 0xcc, 0xcc, 0xcc, 0xcc,
                     0xcc, 0xec, 0x3f, 0x20, 0x01, 0x38, 0x01};

Bytes with_tail(Bytes head) {
  head.insert(head.end(), kTail.begin(), kTail.end());
  return head;
}

// Minimal protobuf wire-format reader for the fields written by
// serialize_jax_session_config.
struct Reader {
  const Bytes& data;
  std::size_t pos = 0;

  bool done() const { return pos >= data.size(); }

  std::uint64_t varint() {
    std::uint64_t value = 0;
    int shift = 0;
    while (true) {
      EXPECT_LT(pos, data.size());
      if (pos >= data.size()) {
        return value;
      }
      const std::uint8_t byte = data[pos++];
      value |= static_cast<std::uint64_t>(byte & 0x7fu) << shift;
      if ((byte & 0x80u) == 0) {
        return value;
      }
      shift += 7;
    }
  }
};

struct Decoded {
  std::uint64_t intra = 0;
  std::uint64_t inter = 0;
  double gpu_memory_fraction = 0.0;
  std::uint64_t allow_growth = 0;
  std::uint64_t allow_soft_placement = 0;
};

Decoded decode(const Bytes& config) {
  Decoded out;
  Reader reader{config};
  while (!reader.done()) {
    const std::uint64_t key = reader.varint();
    const std::uint64_t field = key >> 3;
    const std::uint64_t wire = key & 0x7u;
    if (field == 2 && wire == 0) {
      out.intra = reader.varint();
    } else if (field == 5 && wire == 0) {
      out.inter = reader.varint();
    } else if (field == 7 && wire == 0) {
      out.allow_soft_placement = reader.varint();
    } else if (field == 6 && wire == 2) {
      const std::size_t length = reader.varint();
      const std::size_t end = reader.pos + length;
      EXPECT_LE(end, config.size());
      while (reader.pos < end && reader.pos < config.size()) {
        const std::uint64_t sub_key = reader.varint();
        if (sub_key == ((1u << 3) | 1u)) {
          EXPECT_LE(reader.pos + sizeof(double), end);
          std::memcpy(&out.gpu_memory_fraction, config.data() + reader.pos,
                      sizeof(double));
          reader.pos += sizeof(double);
        } else if (sub_key == ((4u << 3) | 0u)) {
          out.allow_growth = reader.varint();
        } else {
          ADD_FAILURE() << "unexpected gpu_options key " << sub_key;
          return out;
        }
      }
      EXPECT_EQ(reader.pos, end);
    } else {
      ADD_FAILURE() << "unexpected ConfigProto key " << key;
      return out;
    }
  }
  return out;
}

}  // namespace

TEST(JaxSessionConfig, MatchesProtobufBytes) {
  const std::vector<std::pair<std::pair<int, int>, Bytes>> cases = {
      {{0, 0}, with_tail({0x10, 0x00, 0x28, 0x00})},
      {{127, 127}, with_tail({0x10, 0x7f, 0x28, 0x7f})},
      {{128, 128}, with_tail({0x10, 0x80, 0x01, 0x28, 0x80, 0x01})},
      {{255, 2}, with_tail({0x10, 0xff, 0x01, 0x28, 0x02})},
      {{256, 256}, with_tail({0x10, 0x80, 0x02, 0x28, 0x80, 0x02})},
      {{16384, 3}, with_tail({0x10, 0x80, 0x80, 0x01, 0x28, 0x03})},
  };
  for (const auto& item : cases) {
    const int intra = item.first.first;
    const int inter = item.first.second;
    EXPECT_EQ(deepmd::serialize_jax_session_config(intra, inter, 0.9),
              item.second)
        << "intra=" << intra << " inter=" << inter;
  }
}

TEST(JaxSessionConfig, DecodesThreadCountsAndOptions) {
  for (const int intra : {0, 1, 127, 128, 255, 256, 1000, 16384, 65535}) {
    for (const int inter : {0, 1, 127, 128, 300}) {
      const Bytes config =
          deepmd::serialize_jax_session_config(intra, inter, 0.9);
      const Decoded decoded = decode(config);
      EXPECT_EQ(decoded.intra, static_cast<std::uint64_t>(intra));
      EXPECT_EQ(decoded.inter, static_cast<std::uint64_t>(inter));
      EXPECT_DOUBLE_EQ(decoded.gpu_memory_fraction, 0.9);
      EXPECT_EQ(decoded.allow_growth, 1u);
      EXPECT_EQ(decoded.allow_soft_placement, 1u);
    }
  }
}
