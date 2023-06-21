# Comparison with lean-gym 

Here we compare LeanDojo with [lean-gym](https://github.com/openai/lean-gym) on the number of correct proofs misjudged as incorrect (details in Appendix A.2 of our paper).


## Install lean-gym

```bash
git clone https://github.com/openai/lean-gym
cd lean-gym
git checkout 2ebaf0dc909c7bcb03a8477a708345aea6e13dc4
sed -i 's/b72300f3455ae73c3ab9ed40fc1f80bbb9c85ba4/6e5ca7d0097313e59f7533a42e3ea5197484c775/' leanpkg.toml
sed -i 's/lean:3.39.1/lean:3.42.1/' leanpkg.toml
sed -i s'/5000/600000/g' lean-gym/src/repl.lean
bash ./scripts/setup.sh
```


## Run the Comparison

```bash
python evaluate_interaction_tools.py
```