import numpy as np
# import click

def read_inputs(file):

    lines = []
    with open(f'./testcases/{file}.txt', 'r') as fin:
        
        for line in fin:
            lines.append(line.rstrip())

    h_val = float(lines[0])
    iterations = int(lines[1])
    # breakpoint()

    nodes = set()

    vsrc_components = []
    isrc_components = []
    r_components = []
    c_components = []
    i_components = []

    idx = 2
    while(idx < len(lines)):
        arr = lines[idx].split()
        if len(arr) != 5:
            break
    
        if arr[0] == 'Vsrc':
            vsrc_components.append(arr)      
        elif arr[0] == 'Isrc':
            isrc_components.append(arr)      
        elif arr[0] == 'R':
            r_components.append(arr)
        elif arr[0] == 'C':
            c_components.append(arr)
        elif arr[0] == 'I':
            i_components.append(arr)

        nodes.add(arr[1])    
        nodes.add(arr[2])
        idx += 1

    return nodes, h_val, iterations, vsrc_components, isrc_components, r_components, c_components, i_components


def G_matrix(r_components, c_components, n, m, h):
    G = np.zeros((n, n))

    for idx, comp in enumerate(r_components):
        
        node1 = int(comp[1][1]) - 1
        node2 = int(comp[2][1]) - 1
        r_val = float(comp[3])

        if node1 >= 0:
            G[node1][node1] += (1/r_val)
        if node2 >= 0:
            G[node2][node2] += (1/r_val)
        if node1 >= 0 and node2 >= 0:
            G[node1][node2] -= (1/r_val)
            G[node2][node1] -= (1/r_val)

    for idx, comp in enumerate(c_components):
        
        node1 = int(comp[1][1]) - 1
        node2 = int(comp[2][1]) - 1
        c_val = float(comp[3])

        if node1 >= 0:
            G[node1][node1] += (c_val/h)
        if node2 >= 0:
            G[node2][node2] += (c_val/h)
        if node1 >= 0 and node2 >= 0:
            G[node1][node2] -= (c_val/h)
            G[node2][node1] -= (c_val/h)

    return G


def B_matrix(vsrc_components, i_components, prev_x, n, m, h):
    B = np.zeros((n, m))

    for idx, comp in enumerate(vsrc_components):

        # Positive
        node1 = int(comp[1][1]) - 1
        # Negative 
        node2 = int(comp[2][1]) - 1

        if node1 >= 0:
            B[node1][idx] = 1
        if node2 >= 0:
            B[node2][idx] = -1
    
    for idx, comp in enumerate(i_components):

        inductor_idx = idx+len(vsrc_components)
        node1 = int(comp[1][1]) - 1
        node2 = int(comp[2][1]) - 1

        if node1 >= 0:
            B[node1][inductor_idx] = 1
        if node2 >= 0:
            B[node2][inductor_idx] = -1
            
    return B


def D_matrix(vsrc_components, i_components, n, m, h):
    D = np.zeros((m, m))

    for idx, comp in enumerate(i_components):

        inductor_idx = idx + len(vsrc_components)
        L = float(comp[3])
        D[inductor_idx][inductor_idx] -= L/h

    return D


def Z_vector(vsrc_components, i_components, isrc_components, c_components, prev_x, n, m, h):

    i_vector = np.zeros((n, 1))
    e_vector = np.zeros((m, 1))

    # i_vector
    for idx, comp in enumerate(isrc_components):
        node1 = int(comp[1][1]) - 1
        node2 = int(comp[2][1]) - 1
        isrc_val = float(comp[3])

        if node1 >= 0:
            i_vector[node1][0] += isrc_val

    for idx, comp in enumerate(c_components):
        node1 = int(comp[1][1]) - 1
        node2 = int(comp[2][1]) - 1
        c_val = float(comp[3])

        if node1 < 0:
            delta_v = 0 - prev_x[node1][0]
        elif node2 < 0:
            delta_v = prev_x[node1][0]
        else:
            delta_v = prev_x[node1][0] - prev_x[node2][0]

        if node1 >= 0:
            i_vector[node1][0] += (c_val/h) * delta_v
        if node2 >= 0:
            i_vector[node2][0] -= (c_val/h) * delta_v
        
    # e_vector
    for idx, comp in enumerate(vsrc_components):

        v_val = float(comp[3])
        e_vector[idx][0] = v_val

    for idx, comp in enumerate(i_components):

        inductor_idx = idx+len(vsrc_components)
        L = float(comp[3])
        e_vector[inductor_idx][0] -= (L/h) * prev_x[n + len(vsrc_components) + idx][0]

    return np.block([[i_vector], [e_vector]])


def format(results, n, n_v, m):

    v_nodes = [[] for i in range(n)]
    current_nodes = [[] for i in range(m)]

    for result in results:
        (h_val, x) = result
        for i in range(n):
            v_nodes[i].append(f'{round(h_val, 1)} {round(x[i][0], 10)}')
        for i in range(m):
            current_nodes[i].append(f'{round(h_val, 1)} {round(x[i+n][0], 10)}')

    with open('results.txt', 'w') as fo:  
        for idx, node in enumerate(v_nodes):
            fo.writelines(f'V{idx+1}\n')
            for res in node:
                fo.writelines(f'{res}\n')
            fo.write('\n')
    
        for idx, node in enumerate(current_nodes):
            if idx < n_v:
                fo.writelines(f'I_Vsrc{idx}\n')
            else:
                fo.writelines(f'I_L{idx-n_v}\n')
            for res in node:
                fo.writelines(f'{res}\n')
            fo.write('\n')
    

# @click.command()
# @click.option('--testfile', prompt='Testcase', help='Name of Testcase file.', required=True)
def compute(testfile):

    nodes, h, iterations, vsrc_components, isrc_components, r_components, c_components, i_components = read_inputs(testfile)
    
    n = len(nodes) - 1
    m = len(vsrc_components) + len(i_components)
    time_step = h

    prev_x = np.zeros((n+m,1))

    results = []
    # Matrices
    G = G_matrix(r_components, c_components, n, m, h)
    B = B_matrix(vsrc_components, i_components, prev_x, n, m, h)
    C = B.T
    D = D_matrix(vsrc_components, i_components, n, m, h)

    # Ax = Z
    A = np.block([[G, B], [C, D]])

    for iter in range(iterations):

        Z = Z_vector(vsrc_components, i_components, isrc_components, c_components, prev_x, n, m, h)        
        x =  np.dot(np.linalg.inv(A), Z)
        
        # breakpoint()
        # print(f'Result for iteration {iter+1}\nX = \n{np.around(x, decimals=10)}')

        results.append((time_step, x))

        prev_x = x
        time_step += h
    # breakpoint()

    format(results, n, len(vsrc_components), m)

if __name__ == "__main__":

    print("Enter number of testcase: ")
    testcase = input()
    compute(testcase)

    print("Check results. file\nPress any key to exit")
    _ = input()