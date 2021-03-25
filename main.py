"""
This project aims at implementing the Simplex method in Linear Programming problem.

Author: YiChong Huang (黄毅翀)
Student_Code: 20S103272
"""

import torch


def simplex():
    """
    for a Optimization problem:
        Max z = c^T * X
        s.t. A * X = b
               X   > 0

    1. read the size of coefficient matrix A
    2. read the values of coefficient matrix A, coefficient in target function c, b
    3. start iteration:
        while max(check_num) > 0:
            i = argmax(check_num[i])
            j = argmin([b[j] / A[j][i] if A[j][i] > 0]) or None if len([b[j] / A[j][i] if A[j][i] > 0]) == 0
            if j is None:
                return None # No optimized solution
            Update A
    :return: optimized solution
    """

    # 1. read the size of coefficient matrix A
    size_A = input("input the size of coefficient matrix A: (like: 2 5)")
    # size_A = "2 5"
    size_A = [int(i) for i in size_A.split()]

    # 2. read the values of coefficient matrix A, coefficient in target function c, b
    A = []
    c = []
    b = []
    print("input the values of coefficient matrix A:")
    print("(like: ")
    # print("\t 1 4 2 1 0")
    # print("\t 1 2 4 0 1)")
    print("\t 4 5 -2 1 0")
    print("\t 1 -2 1 0 1)")

    for i in range(size_A[0]):
            line_A = input()
            A.append([float(i) for i in line_A.split()])
    # A = [[1., 4., 2., 1., 0.],
    #      [1., 2., 4., 0., 1.]]
    A = torch.Tensor(A)

    print("input the values of coefficient in target function c:")
    # print("(like: 6 14 13 0 0)")
    print("(like: -3 2 4 0 0)")

    c = input()
    # c = "6 14 13 0 0"
    c = [float(i) for i in c.split()]
    c = torch.Tensor(c)

    print("input the values of b:")
    # print("(like: 48 60)")
    print("(like: 22 30)")
    b = input()
    # b = "48 60"
    b = [float(i) for i in b.split()]
    b = torch.Tensor(b)

    # 3. start iteration:
    x_B = []  # we assume that there are enough basic value
    for i in range(size_A[1]):
        if [A[j][i] for j in range(size_A[0])].count(1.) == 1 \
                and [A[j][i] for j in range(size_A[0])].count(0.) == size_A[0] - 1:
            x_B.append(i)

    c_B = [c[i] for i in x_B]
    c_B = torch.Tensor(c_B)

    # calculate check number:
    check_num = []
    for i in range(size_A[1]):
        check_num.append(c[i] - c_B.dot(A[:, i]))
    check_num = torch.Tensor(check_num)

    print("---------------------------------------------------------------")

    s_columns = ["x_" + str(i+1) for i in range(size_A[1])]
    s_columns = "\t\t".join(s_columns)
    print("c_B\tx_B\tb\t\t" + s_columns)
    for i in range(size_A[0]):
        print("{}\t{}\t{}\t{}".format(c_B[i], "x_"+str(x_B[i]+1), b[i], "\t\t".join([str(j) for j in A[i].tolist()])))
    print("check_num:\t\t", "\t".join([str(j.item()) for j in check_num]))

    while max(check_num) > 0:
        new_basic_index = check_num.argmax().item()
        theta = []
        end_flag = True
        for i in range(size_A[0]):
            if A[i][new_basic_index] > 0:
                end_flag = False
                theta.append(b[i] / A[i][new_basic_index])
            else:
                theta.append(10000000)
        if end_flag:
            print("No Optimized Solution!!!")
            return 0
        replace_basic_index = torch.Tensor(theta).argmin().item()

        print("new_basic_index", new_basic_index)
        print("replace_basic_index", replace_basic_index)

        # update table:
        # 1) update x_B
        x_B[replace_basic_index] = new_basic_index

        # 2) update c_B
        c_B = [c[i] for i in x_B]
        c_B = torch.Tensor(c_B)

        # 3) update A and b
        b[replace_basic_index] = b[replace_basic_index].div(A[replace_basic_index, new_basic_index])
        A[replace_basic_index] = A[replace_basic_index].div(A[replace_basic_index, new_basic_index])
        for i in range(size_A[0]):
            if i != replace_basic_index and A[i][new_basic_index] != 0.:
                times = A[i][new_basic_index].item()
                A[i] -= times * A[replace_basic_index]
                b[i] -= times * b[replace_basic_index]

        # calculate check number:
        check_num = []
        for i in range(size_A[1]):
            check_num.append(c[i] - c_B.dot(A[:, i]))
        check_num = torch.Tensor(check_num)

        # print intermediate result
        print("---------------------------------------------------------------")
        for i in range(size_A[0]):
            print("{}\t{}\t{}\t{}".format(c_B[i].item(), "x_" + str(x_B[i] + 1), round(b[i].item(), 3),
                                          "\t".join([str(round(j, 3)) for j in A[i].tolist()])))
        print("check_num:\t", "\t".join([str(round(j.item(), 3)) for j in check_num]))

    print("==================================================================")
    print("Optimal Solution:")
    for i in range(size_A[0]):
        print("x_{} = {}".format(str(x_B[i] + 1), str(round(b[i].item(), 3))))

    # Calculate Optimal value:
    optimal_z = 0
    for i in range(size_A[0]):
        optimal_z += b[i] * c[x_B[i]]
    print("z = {}".format(str(optimal_z.item())))
    return 0


if __name__ == "__main__":
    simplex()