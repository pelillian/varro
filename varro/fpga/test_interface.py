from interface import FpgaConfig

testConfig = FpgaConfig()
print("Attempting to send data [0, 0, 0, 0, 0, 0] to arduino")
results = testConfig.evaluate([0, 0, 0, 0, 0, 0])

print("Contents of results:")
print(" ".join(results))
