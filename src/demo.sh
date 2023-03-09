# python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=1 --epochs=10
python federated_main.py --model=cnn --dataset=mnist --gpu=0 --iid=0 --ka_ep=20 --trainer ka --num_users 10 --frac 1 --local_ep 10

# python federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --ka_ep=5 --trainer ka --num_users 10 --frac 1 --local_ep 10
# python federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10 --trainer ka --num_users 1 --frac 0.1
