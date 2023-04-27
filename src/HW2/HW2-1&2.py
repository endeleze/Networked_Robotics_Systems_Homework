import numpy as np
import argparse
import yaml

class Examiner():
    def __init__(self,question='1',**config):
        self.question=question
        self.matrix={k:np.array(m) for k,m in config.items()}

    def exam(self,k,matrix):
        check=True
        direct=False

        w, v = np.linalg.eig(matrix)

        r = []
        for x in w:
            if isinstance(x, complex):
                direct = True
                r.append(round(x.real, 2))
        r=np.array(r)
        # Check if the matrix is square
        if self.question=='1':
            if matrix.shape[0] != matrix.shape[1]:
                print(f'{k} cannot be the Laplacian of any graph, because the matrix is not square\n')
                return None,None,False,False

            # Check if the column sums are zero
            if not np.all(np.sum(matrix, axis=0) == 0):
                print(f'{k} cannot be the Laplacian of any graph, because some sum of columns are nonzero\n')
                check=False

            # Check if the all real part of eigenvalue are nonzero
            if not np.all(r>=0) and check:
                print(f'{k} cannot be the Laplacian of any graph, because some eigenvalues are negative\n')
                check = False

        # If all tests pass, the matrix is a valid Laplacian
        return w.round(2), v.round(2), direct, check

    def property(self):
        for n,m in self.matrix.items():
            eigenvalue,eigenvector,direct,valid=self.exam(n,m)
            tpye='directed' if direct else 'undirected'
            if self.question=='1':
                if valid:
                    print(n,f'can be the Laplacian of {tpye} graph, and the eigen value is \n{eigenvalue}\n')
            elif self.question=='2':
                print(f'Matrix {n}:\nEigenvalues are:\n{eigenvalue}\nEigenvectors are:\n{eigenvector}\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', type=str, default='2')
    parser.add_argument('-c', '--config', type=str, default='../../input/HW2/2.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    examiner=Examiner(question=args.question,**config)
    examiner.property()
