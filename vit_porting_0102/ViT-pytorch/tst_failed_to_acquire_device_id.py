import torch
import poptorch
import os, time
import pdb


if __name__=='__main__':
    pdb.set_trace()
    model = torch.nn.Linear(5, 2)
    infer_model = poptorch.inferenceModel(model)
    input1 = torch.ones(5)
    print(f'Attaching in {os.getpid()}')
    infer_model.compile(input1)

    use_fork = int(os.environ.get('USE_FORK', '0')) != 0
    print(f'use_fork={use_fork}')

    if use_fork and os.fork() == 0:
        print(f'Start sleep in {os.getpid()}')
        # time.sleep(10)
        time.sleep(300)
    else:
        print(f'Detaching in {os.getpid()}')
        infer_model.detachFromDevice()
        print(f'Attaching in {os.getpid()}')
        infer_model.attachToDevice()

    print(f'Exiting {os.getpid()}')
