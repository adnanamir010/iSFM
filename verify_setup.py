print("--- Running Final NumPy -> Eigen Verification ---")

try:
    import numpy as np
    import hybrid_sfm
    print("[SUCCESS] Packages imported.")

    print("\nCreating a NumPy array for our 2D point...")
    numpy_point = np.array([123.45, 678.90], dtype=np.float64)
    print(f"[SUCCESS] NumPy array created: {numpy_point}")

    print("\nCreating an Observation object...")
    obs = hybrid_sfm.Observation()
    print("[SUCCESS] Observation object created.")

    print("\nAttempting to assign NumPy array to the C++ Eigen member...")
    obs.point = numpy_point  # This triggers the automatic conversion
    print("[SUCCESS] Assignment worked! The NumPy->Eigen bridge is solid.")

    print("\nAttempting to read the value back from C++...")
    # When we read it back, pybind11 automatically converts the Eigen vector
    # back into a new NumPy array for us.
    retrieved_point = obs.point
    print(f"[SUCCESS] Retrieved value is: {retrieved_point}")
    print(f"[SUCCESS] Type of retrieved value is: {type(retrieved_point)}")

    assert np.allclose(numpy_point, retrieved_point)
    print("[SUCCESS] Original and retrieved values match!")

    print("\n\n==================================================")
    print("               THE WAR IS OVER")
    print("          DAY 1 IS OFFICIALLY COMPLETE")
    print("==================================================")

except Exception as e:
    print(f"\n[FATAL] An unexpected error occurred: {e}")
