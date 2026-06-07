# Results

1. Mnist

    ```bash
    (ai_lab) stu_czs37@zxc-N-A:~/projects/class6$ python mnist_attack.py 
    CUDA Available:  True
    100%|██████████████████████████████| 10000/10000 [00:38<00:00, 258.54it/s]
    Epsilon: 0      Test Accuracy = 98.10%
    100%|██████████████████████████████| 10000/10000 [00:38<00:00, 262.98it/s]
    Epsilon: 0.05   Test Accuracy = 94.26%
    100%|██████████████████████████████| 10000/10000 [00:38<00:00, 261.05it/s]
    Epsilon: 0.1    Test Accuracy = 85.09%
    100%|██████████████████████████████| 10000/10000 [00:38<00:00, 262.50it/s]
    Epsilon: 0.15   Test Accuracy = 68.28%
    100%|██████████████████████████████| 10000/10000 [00:38<00:00, 260.86it/s]
    Epsilon: 0.2    Test Accuracy = 43.02%
    100%|██████████████████████████████| 10000/10000 [00:37<00:00, 263.66it/s]
    Epsilon: 0.25   Test Accuracy = 20.81%
    100%|██████████████████████████████| 10000/10000 [00:38<00:00, 259.78it/s]
    Epsilon: 0.3    Test Accuracy = 8.71%
    ```

2. Fool_box

    ```bash
    (ai_lab) stu_czs37@zxc-N-A:~/projects/class6$ python foolbox_practice.py 
    原始精度:  87.5 %
    攻击后的精度：
    Linf norm ≤ 0     : 50.0 %
    Linf norm ≤ 1     : 43.8 %
    Linf norm ≤ 2     : 25.0 %
    Linf norm ≤ 3     : 18.8 %
    Linf norm ≤ 4     :  6.2 %
    Linf norm ≤ 5     :  0.0 %
    防御后的精度：
    Linf norm ≤ 0.0008: 75.0 %
    Linf norm ≤ 0.001 : 75.0 %
    Linf norm ≤ 0.0015: 75.0 %
    Linf norm ≤ 0.002 : 75.0 %
    Linf norm ≤ 0.003 : 68.8 %
    Linf norm ≤ 0.01  : 68.8 %
    ```

3. Mnist_target

    ```bash
    (ai_lab) stu_czs37@zxc-N-A:~/projects/class6$ python mnist_attack_target.py 
    CUDA Available:  True
    100%|██████████████| 10000/10000 [00:39<00:00, 253.29it/s]
    Epsilon: 0      attack success rate = 0.42%
    100%|██████████████| 10000/10000 [00:38<00:00, 260.57it/s]
    Epsilon: 0.05   attack success rate = 1.53%
    100%|██████████████| 10000/10000 [00:39<00:00, 256.18it/s]
    Epsilon: 0.1    attack success rate = 4.54%
    100%|██████████████| 10000/10000 [00:50<00:00, 199.75it/s]
    Epsilon: 0.15   attack success rate = 12.91%
    100%|██████████████| 10000/10000 [01:09<00:00, 143.68it/s]
    Epsilon: 0.2    attack success rate = 32.05%
    100%|██████████████| 10000/10000 [01:25<00:00, 116.60it/s]
    Epsilon: 0.25   attack success rate = 56.78%
    100%|██████████████| 10000/10000 [00:42<00:00, 233.85it/s]
    Epsilon: 0.3    attack success rate = 75.52%
    ```

4. mnist

    ```bash
    ai_lab) stu_czs37@zxc-N-A:~/projects/class6$ python mnist_attack.py 
    CUDA Available:  True
    100%|██████████████| 10000/10000 [00:39<00:00, 255.83it/s]
    Epsilon: 0      Test Accuracy = 98.10%
    100%|██████████████| 10000/10000 [00:37<00:00, 263.88it/s]
    Epsilon: 0.05   Test Accuracy = 94.26%
    100%|██████████████| 10000/10000 [00:43<00:00, 230.66it/s]
    Epsilon: 0.1    Test Accuracy = 85.09%
    100%|██████████████| 10000/10000 [00:53<00:00, 187.25it/s]
    Epsilon: 0.15   Test Accuracy = 68.28%
    100%|██████████████| 10000/10000 [01:20<00:00, 124.99it/s]
    Epsilon: 0.2    Test Accuracy = 43.02%
    100%|██████████████| 10000/10000 [01:34<00:00, 105.34it/s]
    Epsilon: 0.25   Test Accuracy = 20.81%
    100%|██████████████| 10000/10000 [00:38<00:00, 256.55it/s]
    Epsilon: 0.3    Test Accuracy = 8.71%
    ```

4. foolbox(PGD)

```bash
(ai_lab) stu_czs37@zxc-N-A:~/projects/class6$ python foolbox_practice.py 
原始精度:  87.5 %
攻击后的精度：
  Linf norm ≤ 0     : 50.0 %
  Linf norm ≤ 1     : 43.8 %
  Linf norm ≤ 2     : 25.0 %
  Linf norm ≤ 3     : 18.8 %
  Linf norm ≤ 4     :  0.0 %
  Linf norm ≤ 5     :  0.0 %
防御后的精度：
  Linf norm ≤ 0.0008: 68.8 %
  Linf norm ≤ 0.001 : 56.2 %
  Linf norm ≤ 0.0015: 37.5 %
  Linf norm ≤ 0.002 : 37.5 %
  Linf norm ≤ 0.003 : 12.5 %
  Linf norm ≤ 0.01  :  0.0 %
```

6. foolbox(BIM)
```bash

```