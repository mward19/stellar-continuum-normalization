The class BOSZSpectra processes astronomical spectra, and currently has some functionality to cache spectra. However, I'm on a system with a file quantity quota. I'm near the maximum, and I have like 600,000 spectra to work with. I want to store the processed spectra more efficiently as `.npy` files where each file contains some relatively large quantity of spectra, perhaps 100 or 1000 spectra per file. They should be able to be loaded with `data = np.load("processed_000000000.npy", mmap_mode="r"); item = data[123]` so that accessing just one item is efficient. 

I want you to modify the caching system in BOSZSpectra. Give the class another __init__ argument called `spectra_per_file` which gives the number of spectra that each cache npy file should contain. The cache files should go into `cache_dir`, and each should be called `processed_<index>.npy`, where there are nine zeros for the first one (`processed_000000000.npy`). The caching system should create a `.sqlite` database called `spectra_index.sqlite` in `cache_dir`, which has the following columns:
- the original file name
- the spectrum id fields (should be unique when combined)
- the cached `npy` filepath
- this spectrum's row index within in the cached `processed_<index>.npy` file

Store the parameters used in the processing as a separate table in `spectra_index.sqlite` and make new initializations of BOSZSpectra check that the parameters match if the index file already exists.

Partial last group file is fine.

Write the .npy to a temp name, rename it, then insert the DB rows. This keeps the index from pointing at a missing file.

Then make sure that `is_cached`, `processed_spectrum`, etc. check in _this_ cache for the data.