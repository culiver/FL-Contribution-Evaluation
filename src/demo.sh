# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10 --trainer ka --num_users 100 --frac 0.02 --local_ep 1

python federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10 --trainer ka --num_users 100 --frac 0.02 --local_ep 1
