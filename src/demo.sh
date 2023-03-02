# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10
# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10 --trainer ka --num_users 100 --frac 0.02 --local_ep 1

python federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --ka_ep=2 --trainer ka --num_users 5 --frac 0.2 --local_ep 10 --optimizer adam --lr 2e-4
# python federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10 --trainer ka --num_users 1 --frac 0.1
