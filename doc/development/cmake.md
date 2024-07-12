# Find DeePMD-kit C/C++ library from CMake

After DeePMD-kit C/C++ library is installed, one can find DeePMD-kit from CMake:

```cmake
find_package(DeePMD REQUIRED)
```

Note that you may need to add ${deepmd_root} to the cached CMake variable `CMAKE_PREFIX_PATH`.

To link against the C interface library, using

```cmake
target_link_libraries(some_library PRIVATE DeePMD::deepmd_c)
```

To link against the C++ interface library, using

```cmake
target_link_libraries(some_library PRIVATE DeePMD::deepmd_cc)
```
