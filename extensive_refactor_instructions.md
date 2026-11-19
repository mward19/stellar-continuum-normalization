I want to perform an extensive refactoring of the code in model.py. Right now it's hard to work with, a little bloated. 
Split `ContinuumNormalizer` into four classes:
- `SpectralNMF`. This class should have functionality for filtering BOSZ data and both minibatch and full NMF algorithms, managing the cache and such too.
- `TelluricNMF`. This should be similar to `SpectralNMF`, but it doesn't really need the filtering functionality, I don't think.
- `ContinuumResponseBasis`. Constructs the Fourier basis for the matrix $H$. Later I want to try other bases, so keep it flexible.
- `ContinuumNormalizer`. Puts those pieces together to perform weighted least squares as described in the paper.

I am wondering where I should put the resolution/number of samples parameters. Ideally I could perform an NMF at maximum resolution and downsample the basis vectors to the dimensionality of my data, right?
