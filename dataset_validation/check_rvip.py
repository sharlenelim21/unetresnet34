import nrrd
import numpy as np

# Install if needed: pip install pynrrd
data, header = nrrd.read('data/rvip/patient001_frame01_rvip.nrrd')

print("Shape:", data.shape)
print("Unique values:", np.unique(data))
print("Header:", header)
print("Non-zero pixels:", np.sum(data > 0))