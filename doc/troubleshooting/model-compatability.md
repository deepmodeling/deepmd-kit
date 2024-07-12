# Model compatibility

When the version of DeePMD-kit used to train the model is different from the that of DeePMD-kit running MDs, one has the problem of model compatibility.

DeePMD-kit guarantees that the codes with the same major and minor revisions are compatible. That is to say, v0.12.5 is compatible with v0.12.0, but is not compatible with v0.11.0 or v1.0.0.

One can execute `dp convert-from` to convert an old model to a new one.

| Model version | v0.12 | v1.0 | v1.1 | v1.2 | v1.3 | v2.0 | v2.1 | v2.2 |
| :-----------: | :---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Compatibility |  😊   |  😊  |  😊  |  😊  |  😊  |  😄  |  😄  |  😄  |

**Legend**:

- 😄: The model is compatible with the DeePMD-kit package.
- 😊: The model is incompatible with the DeePMD-kit package, but one can execute `dp convert-from` to convert an old model to v2.2.
- 😢: The model is incompatible with the DeePMD-kit package, and there is no way to convert models.
