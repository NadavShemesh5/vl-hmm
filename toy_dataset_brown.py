from dataset_creation_brown import load_dataset
from hmmlearn import hmm_brown

datasets = load_dataset()
X_train = datasets["train"]
X_test = datasets["test"]
X_valid = datasets["valid"]
token2cluster_arr = datasets["vocab"]["token2cluster_arr"]

model = hmm_brown.CategoricalHMM(
    n_states=16384,
    n_iter=100,
    implementation="scaling",
    random_state=5,
    # startprob_prior=0.0001,
    # transmat_prior=0.0001,
    # emissionprob_prior=0.0001,
)


model.fit(
    X_train["tokens"].reshape(-1, 1),
    lengths=X_train["lengths"].reshape(-1, 1),
    token2cluster=token2cluster_arr,
    valid=X_valid["tokens"].reshape(-1, 1),
    valid_lengths=X_valid["lengths"].reshape(-1, 1),
)


perplexity = model.perplexity(
    X_test["tokens"].reshape(-1, 1), lengths=X_test["lengths"].reshape(-1, 1)
)
print(f"Perplexity: {perplexity}")
