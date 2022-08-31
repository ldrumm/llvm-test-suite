// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -I . -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// disabling hip because some of the binary_ops tested are not supported
// getting undefined symbols for a handful of __spirv__ * functions.
// XFAIL: hip

#include "support.h"
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <limits>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

template <class SpecializationKernelName, int TestNumber>
class inclusive_scan_kernel;

// std::inclusive_scan isn't implemented yet, so use serial implementation
// instead
namespace emu {
template <typename InputIterator, typename OutputIterator,
          class BinaryOperation, typename T>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, BinaryOperation binary_op,
                              T init) {
  T partial = init;
  for (InputIterator it = first; it != last; ++it) {
    partial = binary_op(partial, *it);
    *(result++) = partial;
  }
  return result;
}
} // namespace emu

template <typename T> bool approx_eq(T a, T b) {
  return std::abs(a - b) / (0.5 * std::abs(a + b)) < 1e-6;
}

template <> bool approx_eq<int>(int a, int b) { return a == b; }

template <typename InputContainer, typename OutputContainer,
          class BinaryOperation>
void test(queue q, InputContainer input, OutputContainer output,
          BinaryOperation binary_op,
          typename OutputContainer::value_type identity) {
  typedef typename InputContainer::value_type InputT;
  typedef typename OutputContainer::value_type OutputT;
  OutputT init = OutputT(42);
  size_t N = input.size();
  size_t G = 64;
  std::vector<OutputT> expected(N);
  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[lid] = inclusive_scan_over_group(g, in[lid], binary_op);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + G, expected.begin(),
                      binary_op, identity);
  assert(std::equal(output.begin(), output.begin() + G, expected.begin(),
                    approx_eq<OutputT>));

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        int lid = it.get_local_id(0);
        out[lid] = inclusive_scan_over_group(g, in[lid], binary_op, init);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + G, expected.begin(),
                      binary_op, init);
  assert(std::equal(output.begin(), output.begin() + G, expected.begin(),
                    approx_eq<OutputT>));

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        joint_inclusive_scan(g, in.get_pointer(), in.get_pointer() + N,
                             out.get_pointer(), binary_op);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + N, expected.begin(),
                      binary_op, identity);
  assert(std::equal(output.begin(), output.begin() + N, expected.begin(),
                    approx_eq<OutputT>));

  {
    buffer<InputT> in_buf(input.data(), input.size());
    buffer<OutputT> out_buf(output.data(), output.size());
    q.submit([&](handler &cgh) {
      accessor in{in_buf, cgh, sycl::read_only};
      accessor out{out_buf, cgh, sycl::write_only, sycl::no_init};
      cgh.parallel_for(nd_range<1>(G, G), [=](nd_item<1> it) {
        group<1> g = it.get_group();
        joint_inclusive_scan(g, in.get_pointer(), in.get_pointer() + N,
                             out.get_pointer(), binary_op, init);
      });
    });
  }
  emu::inclusive_scan(input.begin(), input.begin() + N, expected.begin(),
                      binary_op, init);
  assert(std::equal(output.begin(), output.begin() + N, expected.begin(),
                    approx_eq<OutputT>));
}

int main() {
  queue q;
  if (!isSupportedDevice(q.get_device())) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 128;
  std::array<int, N> input;
  std::array<int, N> output;
  std::iota(input.begin(), input.end(), 0);
  std::fill(output.begin(), output.end(), 0);

  test(q, input, output, sycl::plus<>(), 0);
  test(q, input, output, sycl::minimum<>(), std::numeric_limits<int>::max());
  test(q, input, output, sycl::maximum<>(), std::numeric_limits<int>::lowest());

  test(q, input, output, sycl::plus<int>(), 0);
  test(q, input, output, sycl::minimum<int>(), std::numeric_limits<int>::max());
  test(q, input, output, sycl::maximum<int>(),
       std::numeric_limits<int>::lowest());
  test(q, input, output, sycl::multiplies<int>(), 1);
  test(q, input, output, sycl::bit_or<int>(), 0);
  test(q, input, output, sycl::bit_xor<int>(), 0);
  test(q, input, output, sycl::bit_and<int>(), ~0);

  std::array<std::complex<float>, N> cf_input;
  std::array<std::complex<float>, N> cf_output;
  for (int i = 0; i < N; i++) {
    cf_input[i] = {(i % 2) + 1.0f, (i % 3) - 1.f};
  }
  std::fill(cf_output.begin(), cf_output.end(), 0);
  test(q, cf_input, cf_output, sycl::multiplies<std::complex<float>>(), 1);

  std::array<std::complex<double>, N> cd_input;
  std::array<std::complex<double>, N> cd_output;
  for (int i = 0; i < N; i++) {
    cd_input[i] = {(i % 2) + 1.0f, (i % 3) - 1.f};
  }
  std::fill(cd_output.begin(), cd_output.end(), 0);
  test(q, cd_input, cd_output, sycl::multiplies<std::complex<double>>(), 1);

  std::array<std::complex<half>, N> ch_input;
  std::array<std::complex<half>, N> ch_output;
  for (int i = 0; i < N; i++) {
    ch_input[i] = {(i % 2) + 1.0f, (i % 3) - 1.f};
  }
  std::fill(ch_output.begin(), ch_output.end(), half(0));
  test(q, ch_input, ch_output, sycl::multiplies<std::complex<half>>(),
       half(1.f));

  std::cout << "Test passed." << std::endl;
}
