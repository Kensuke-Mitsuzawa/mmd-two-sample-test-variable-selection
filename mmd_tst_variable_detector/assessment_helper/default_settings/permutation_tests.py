from ot import sliced_wasserstein_distance

from ...utils.permutation_test_runner import PermutationTest


permutation_test_runner = PermutationTest(
    func_distance=lambda x, y: sliced_wasserstein_distance(x, y),
    n_permutation_test=5000
)
