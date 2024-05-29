from arguments import JSONParameters
args = JSONParameters()

from annealer.arun import annealing_AutoSearch
from utils.utils import redirect
from utils.data import write_libffm_data, generate_ffm_data
from fm.engine import ffm

import os 
from tqdm import tqdm


if __name__ == '__main__':
    print("Converting DFT data to linffm...")
    libffm_data = write_libffm_data(args.work_path.dft_data, args.work_path.fm)
    ffmModel = ffm()
    composition = args.basic.init_composition
    print("Finished! Start optimization!")
    print("Total iteration: ", args.basic.iterations)
    print("Quantum Annealing Platform: ", args.quantum_annealer.type)
    print("System: ", args.basic.elements)
    for i in tqdm(range(args.basic.iterations)):
        generate_ffm_data(fmpath=args.work_path.fm, sample_ratio=args.factor_machine.sampling_ratio)
        with redirect("xlearn.log"):
            ffmModel.train(trainSet=os.path.join(args.work_path.fm, "train_ffm.txt"),
                           validSet=os.path.join(args.work_path.fm, "valid_ffm.txt"),
                           model_txt=os.path.join(args.work_path.fm, "model.txt"),
                           model_out=os.path.join(args.work_path.fm, "train.model"),
                           restart=True)
        composition = annealing_AutoSearch(initial_composition=composition, 
                                           mix_circle=args.quantum_annealer.mix_circle, annealer=args.quantum_annealer.type)

