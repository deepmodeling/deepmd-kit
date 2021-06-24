# Model compatibility

When the version of DeePMD-kit used to training model is different from the that of DeePMD-kit running MDs, one has the problem of model compatibility.

DeePMD-kit guarantees that the codes with the same major and minor revisions are compatible. That is to say v0.12.5 is compatible to v0.12.0, but is not compatible to v0.11.0 nor v1.0.0. 

One can execuate `dp convert-from` to convert an old model to a new one.

|   | Model v0.12 | Model v1.0 | Model v1.1 | Model v1.2 | Model v1.3 | Model v2.0 |
|:-:|:-----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|DeePMD-kit v0.12 | 😄 | 😢 | 😢 | 😢 | 😢 | 😢 |
|DeePMD-kit v1.0  | 😢 | 😄 | 😢 | 😢 | 😢 | 😢 |
|DeePMD-kit v1.1  | 😢 | 😢 | 😄 | 😢 | 😢 | 😢 |
|DeePMD-kit v1.2  | 😢 | 😢 | 😢 | 😄 | 😢 | 😢 |
|DeePMD-kit v1.3  | 😢 | 😢 | 😢 | 😊 | 😄 | 😢 |
|DeePMD-kit v2.0  | 😢 | 😢 | 😢 | 😊 | 😊 | 😄 |

**Legend**:
- 😄: The model is compatible with the DeePMD-kit package.
- 😊: The model is incompatible with the DeePMD-kit package, but one can execuate `dp convert-from` (in the new version) or `dp convert-to` (in the old version) to convert an model to another one.
- 😢: The model is incompatible with the DeePMD-kit package, and there is no way to convert models.
