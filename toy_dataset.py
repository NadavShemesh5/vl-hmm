from dataset_creation import load_dataset
from hmmlearn import hmm

datasets = load_dataset()
X_train = datasets["train"]
X_test = datasets["test"]
X_valid = datasets["valid"]


model = hmm.CategoricalHMM(
    n_states=16384,
    n_iter=100,
    n_clusters=128,
    implementation="scaling",
    random_state=42,
    dropout_rate=0.5,
)


model.fit(
    X_train["tokens"].reshape(-1, 1),
    lengths=X_train["lengths"].reshape(-1, 1),
    valid=X_valid["tokens"].reshape(-1, 1),
    valid_lengths=X_valid["lengths"].reshape(-1, 1),
)


perplexity = model.perplexity(
    X_test["tokens"].reshape(-1, 1), lengths=X_test["lengths"].reshape(-1, 1)
)
print(f"Perplexity: {perplexity}")

# use the Viterbi algorithm to predict the most likely sequence of states
# given the model
# states = model.predict(X_validate)
# print(states)
#
# pred = (
#     model.predict_proba(X_validate)[-1].reshape(-1, 1).T
#     @ model.transmat_
#     @ model.emissionprob_
# )
# print(np.argmax(pred))
#
# # %%
# # Let's check our learned transition probabilities and see if they match.
#
# print(f"Transmission Matrix Generated:\n{model.transmat_.round(3)}\n\n")
#
# # %%
# # Finally, let's see if we can tell how the die is loaded.
#
# print(f"Emission Matrix Generated:\n{model.emissionprob_.round(3)}\n\n")
