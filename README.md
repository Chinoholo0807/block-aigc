# Block AIGC

```bash
nohup python ./benchmark/diffusionopt.py --device=cuda:0 > nohup/new_opt_seed=0.log &

nohup python ./benchmark/sac.py --device=cuda:1 > nohup/new_sac_seed=0.log &

nohup python ./benchmark/rand.py --device=cuda:1 > nohup/new_rand_seed=0.log &

nohup python ./benchmark/crashavoid.py --device=cuda:1 > nohup/new_crashavoid_seed=0.log &
```
