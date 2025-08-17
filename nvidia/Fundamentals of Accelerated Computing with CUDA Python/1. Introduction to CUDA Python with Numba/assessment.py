
import numpy as np
import cupy as cp
import time

def assess(student_function, args):
    print("Setting n to 100 million.")
    n_large = 100_000_000
    np.random.seed(0)
    greyscales_ref = np.random.randint(0, 256, n_large, dtype=np.uint8)
    weights_ref = np.random.rand(n_large)
    
    def reference_implementation(n, greyscales, weights, **kwargs):
        # Use CuPy for a fast, correct reference implementation
        greyscales_gpu = cp.asarray(greyscales)
        weights_gpu = cp.asarray(weights)
        normalized = greyscales_gpu / 255.0
        weighted = normalized * weights_gpu
        activated = 1.0 / (1.0 + cp.exp(-weighted))
        return cp.asnumpy(activated)
    
    reference_result = reference_implementation(n_large, greyscales_ref, weights_ref)

    args['n'] = n_large
    args['greyscales'] = greyscales_ref
    args['weights'] = weights_ref
    
    start_time = time.time()
    user_result = student_function(**args)
    end_time = time.time()
    duration = end_time - start_time
    
    is_correct = np.allclose(user_result, reference_result)

    print(f"\nYour function returns a host np.ndarray: {isinstance(user_result, np.ndarray)}")
    print(f"Your function took {duration:.2f}s to run.")
    print(f"Your function runs fast enough (less than 1 second): {duration < 1.0}")
    print(f"\nYour function returns the correct results: {is_correct}")

    if not is_correct:
        print("Your function is not returning the correct result. Please fix and try again.")
    elif not (duration < 1.0):
         print("Please refactor your code to run faster and try again.")
    else:
        print("\nCongratulations! Your code passed the assessment. 🎉")
