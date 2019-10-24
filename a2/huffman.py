"""
Code for compressing and decompressing using Huffman compression.
"""

from nodes import HuffmanNode, ReadNode


# ====================
# Helper functions for manipulating bytes


def get_bit(byte, bit_num):
    """ Return bit number bit_num from right in byte.

    @param int byte: a given byte
    @param int bit_num: a specific bit number within the byte
    @rtype: int

    >>> get_bit(0b00000101, 2)
    1
    >>> get_bit(0b00000101, 1)
    0
    """
    return (byte & (1 << bit_num)) >> bit_num


def byte_to_bits(byte):
    """ Return the representation of a byte as a string of bits.

    @param int byte: a given byte
    @rtype: str

    >>> byte_to_bits(14)
    '00001110'
    """
    return "".join([str(get_bit(byte, bit_num))
                    for bit_num in range(7, -1, -1)])


def bits_to_byte(bits):
    """ Return int represented by bits, padded on right.

    @param str bits: a string representation of some bits
    @rtype: int

    >>> bits_to_byte("00000101")
    5
    >>> bits_to_byte("101") == 0b10100000
    True
    """
    return sum([int(bits[pos]) << (7 - pos)
                for pos in range(len(bits))])


# ====================
# Functions for compression


def make_freq_dict(text):
    """ Return a dictionary that maps each byte in text to its frequency.

    @param bytes text: a bytes object
    @rtype: dict{int,int}

    >>> d = make_freq_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for byte in text:
        if byte not in freq_dict:
            freq_dict[byte] = 1
        else:
            freq_dict[byte] += 1
    return freq_dict


def huffman_tree(freq_dict):
    """ Return the root HuffmanNode of a Huffman tree corresponding
    to frequency dictionary freq_dict.

    @param dict(int,int) freq_dict: a frequency dictionary
    @rtype: HuffmanNode

    >>> freq = {2: 6, 3: 4}
    >>> t = huffman_tree(freq)
    >>> result1 = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> result2 = HuffmanNode(None, HuffmanNode(2), HuffmanNode(3))
    >>> t == result1 or t == result2
    True
    """
    symbol = None
    new_node = HuffmanNode()
    lst_node = []
    for byte in freq_dict:
        node = HuffmanNode(byte, None, None)
        node_tuple = (freq_dict[byte], node)
        lst_node.append(node_tuple)
    lst_node.sort()
    if len(lst_node) == 1:
        root = HuffmanNode(symbol, lst_node[0][1], None)
    else:
        while len(lst_node) > 1:
            freq = lst_node[0][0] + lst_node[1][0]
            left = lst_node[0][1]
            right = lst_node[1][1]
            new_node = HuffmanNode(None, left, right)
            lst_node.pop(0)
            lst_node.pop(0)
            new_node_tuple = (freq, new_node)
            lst_node.append(new_node_tuple)
            lst_node.sort()
        root = HuffmanNode(symbol, new_node.left, new_node.right)
    return root


def get_codes(tree):
    """ Return a dict mapping symbols from tree rooted at HuffmanNode to codes.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: dict(int,str)

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    paths = find_code_path(tree)
    code_dict = {}
    for path in paths:
        temp = ""
        for code in path[:-1]:
            temp += str(code)
        code_dict[path[-1]] = temp
    return code_dict


def find_code_path(tree):
    """Return all path from the tree. Whenever take a left branch, append
    a 0 to the code; whenever take a right branch, append a 1 to the code.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: list of list

    >>> b = {2: 1, 3: 4, 5: 6, 4: 8}
    >>> t = huffman_tree(b)
    >>> a = find_code_path(t)
    >>> a
    [[0, 4], [1, 0, 0, 2], [1, 0, 1, 3], [1, 1, 5]]

    """
    paths = []
    if not tree:
        return paths
    if tree.left is None and tree.right is None:
        paths.append([tree.symbol])
    for paths1 in find_code_path(tree.left):
        paths.append([0] + paths1)
    for paths2 in find_code_path(tree.right):
        paths.append([1] + paths2)
    return paths


def number_nodes(tree):
    """ Number internal nodes in tree according to postorder traversal;
    start numbering at 0.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(None, HuffmanNode(9), HuffmanNode(10))
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    if not tree.left and not tree.right:
        pass
    elif tree.left.is_leaf() and tree.right.is_leaf():
        tree.number = 0
    else:
        get_count_internal(tree, 0)


def get_count_internal(tree, num):
    """Return the number of internal.

    @param HuffmanNode tree:  a Huffman tree rooted at node 'tree'
    @param num: a number begin to count
    @rtype: int
    >>> b = {2: 1, 3: 4, 5: 6, 4: 8}
    >>> t = huffman_tree(b)
    >>> a = get_count_internal(t, 0)
    >>> a
    3
    """
    if tree.left:
        if not tree.left.is_leaf():
            num = get_count_internal(tree.left, num)
    if tree.right:
        if not tree.right.is_leaf():
            num = get_count_internal(tree.right, num)
    tree.number = num
    return num + 1


def avg_length(tree, freq_dict):
    """ Return the number of bits per symbol required to compress text
    made of the symbols and frequencies in freq_dict, using the Huffman tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: float

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(9)
    >>> tree = HuffmanNode(None, left, right)
    >>> avg_length(tree, freq)
    1.9
    """
    total_bits = 0
    total_freq = 0
    code_dict = get_codes(tree)
    for byte in code_dict:
        freq = freq_dict[byte]
        total_freq += freq_dict[byte]
        for code in code_dict[byte]:
            total_bits += len(code) * freq
    return total_bits / total_freq


def generate_compressed(text, codes):
    """ Return compressed form of text, using mapping in codes for each symbol.

    @param bytes text: a bytes object
    @param dict(int,str) codes: mappings from symbols to codes
    @rtype: bytes

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = generate_compressed(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    acc = ''
    for byte in text:
        acc += codes[byte]
    result = []
    for i in range(0, len(acc), 8):
        result.append(bits_to_byte(acc[i: i+8]))
    return bytes(result)


def tree_to_bytes(tree):
    """ Return a bytes representation of the tree rooted at tree.

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes

    The representation should be based on the postorder traversal of tree
    internal nodes, starting from 0.
    Precondition: tree has its nodes numbered.

    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    acc = get_tree_to_bytes(tree)
    return bytes(acc)


def get_tree_to_bytes(tree):
    """Reutrn a list of bytes.
    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: list
    >>> tree = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> number_nodes(tree)
    >>> get_tree_to_bytes(tree)
    [0, 3, 0, 2]
    >>> left = HuffmanNode(None, HuffmanNode(3), HuffmanNode(2))
    >>> right = HuffmanNode(5)
    >>> tree = HuffmanNode(None, left, right)
    >>> number_nodes(tree)
    >>> get_tree_to_bytes(tree)
    [0, 3, 0, 2, 1, 0, 0, 5]
    """
    result = []
    if tree.left and tree.right:
        result.extend(get_tree_to_bytes(tree.left))
        result.extend(get_tree_to_bytes(tree.right))
        if not tree.left.left and not tree.left.right:
            result.append(0)
            result.append(tree.left.symbol)
        else:
            result.append(1)
            result.append(tree.left.number)
        if not tree.right.left and not tree.right.right:
            result.append(0)
            result.append(tree.right.symbol)
        else:
            result.append(1)
            result.append(tree.right.number)
        if not tree.left and not tree.right:
            result.extend(get_tree_to_bytes(tree.left))
            result.extend(get_tree_to_bytes(tree.right))
    return result


def num_nodes_to_bytes(tree):
    """ Return number of nodes required to represent tree (the root of a
    numbered Huffman tree).

    @param HuffmanNode tree: a Huffman tree rooted at node 'tree'
    @rtype: bytes
    """
    return bytes([tree.number + 1])


def size_to_bytes(size):
    """ Return the size as a bytes object.

    @param int size: a 32-bit integer that we want to convert to bytes
    @rtype: bytes

    >>> list(size_to_bytes(300))
    [44, 1, 0, 0]
    """
    # little-endian representation of 32-bit (4-byte)
    # int size
    return size.to_bytes(4, "little")


def compress(in_file, out_file):
    """ Compress contents of in_file and store results in out_file.

    @param str in_file: input file whose contents we want to compress
    @param str out_file: output file, where we store our compressed result
    @rtype: NoneType
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = make_freq_dict(text)
    tree = huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (num_nodes_to_bytes(tree) + tree_to_bytes(tree) +
              size_to_bytes(len(text)))
    result += generate_compressed(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression


def generate_tree_general(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes nothing about the order of the nodes in the list.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(10, None, None), \
HuffmanNode(12, None, None)), \
HuffmanNode(None, HuffmanNode(5, None, None), HuffmanNode(7, None, None)))
    """
    result = HuffmanNode()
    if node_lst[root_index].l_type == 0:
        lsymbol = node_lst[root_index].l_data
        result.left = HuffmanNode(lsymbol, None, None)
    else:
        result.left = generate_tree_general(node_lst, node_lst[
            root_index].l_data)
    if node_lst[root_index].r_type == 0:
        symbol = node_lst[root_index].r_data
        result.right = HuffmanNode(symbol, None, None)
    else:
        result.right = generate_tree_general(node_lst, node_lst[
            root_index].r_data)
    return result


def count_internal(node_lst):
    """Return the number of internal from the node_lst.

     @param list[ReadNode] node_lst: a list of ReadNode objects
     @rtype: int

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 2, 1, 0), ReadNode(1, 0, 1, 0)]
    >>> count_internal(lst)
    2

    """
    acc = 0
    index = len(node_lst) - 1
    if index >= 0:
        while node_lst[index].l_type != 0 or node_lst[index].r_type != 0:
            acc += 1
            index -= 1
        return acc


def generate_tree_postorder(node_lst, root_index):
    """ Return the root of the Huffman tree corresponding
    to node_lst[root_index].

    The function assumes that the list represents a tree in postorder.

    @param list[ReadNode] node_lst: a list of ReadNode objects
    @param int root_index: index in the node list
    @rtype: HuffmanNode

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanNode(None, HuffmanNode(None, HuffmanNode(5, None, None), \
HuffmanNode(7, None, None)), \
HuffmanNode(None, HuffmanNode(10, None, None), HuffmanNode(12, None, None)))
    """
    result = HuffmanNode()
    c = (root_index + 1) - (count_internal(node_lst) + 1)
    if node_lst[root_index].l_type == 0:
        if node_lst[root_index].r_type == 0:
            lsymbol = node_lst[root_index].l_data
            rsymbol = node_lst[root_index].r_data
            result.left = HuffmanNode(lsymbol, None, None)
            result.right = HuffmanNode(rsymbol, None, None)
    if node_lst[root_index].l_type != 0:
        if node_lst[root_index].r_type == 0:
            result.left = generate_tree_postorder(node_lst[0:-2], node_lst[
                root_index].l_data)
            rsymbol = node_lst[-2].r_data
            result.right = HuffmanNode(rsymbol, None, None)
    if node_lst[root_index].r_type != 0:
        if node_lst[root_index].l_type == 0:
            lsymbol = node_lst[0].l_data
            result.left = HuffmanNode(lsymbol, None, None)
            result.right = generate_tree_postorder(node_lst[1:-1], node_lst[
                root_index].r_data)
    if node_lst[root_index].r_type != 0:
        if node_lst[root_index].l_type != 0:
            result.right = generate_tree_postorder(node_lst[c:-1], node_lst[
                root_index].r_data)
            result.left = generate_tree_postorder(node_lst[0:c], node_lst[
                root_index].l_data)
    return result


def generate_uncompressed(tree, text, size):
    """ Use Huffman tree to decompress size bytes from text.

    @param HuffmanNode tree: a HuffmanNode tree rooted at 'tree'
    @param bytes text: text to decompress
    @param int size: how many bytes to decompress from text.
    @rtype: bytes
    """
    acc = ''
    result = []
    for i in text:
        acc += byte_to_bits(i)
    t = tree
    for num in acc:
        # check tree exist
        if t:
            if num == '0':
                t = t.left
            if num == '1':
                t = t.right
            if t and t.is_leaf():
                result.append(t.symbol)
                t = tree
                if len(result) == size:
                    return bytes(result)
    return bytes(result)


def bytes_to_nodes(buf):
    """ Return a list of ReadNodes corresponding to the bytes in buf.

    @param bytes buf: a bytes object
    @rtype: list[ReadNode]

    >>> bytes_to_nodes(bytes([0, 1, 0, 2]))
    [ReadNode(0, 1, 0, 2)]
    """
    lst = []
    for i in range(0, len(buf), 4):
        l_type = buf[i]
        l_data = buf[i+1]
        r_type = buf[i+2]
        r_data = buf[i+3]
        lst.append(ReadNode(l_type, l_data, r_type, r_data))
    return lst


def bytes_to_size(buf):
    """ Return the size corresponding to the
    given 4-byte little-endian representation.

    @param bytes buf: a bytes object
    @rtype: int

    >>> bytes_to_size(bytes([44, 1, 0, 0]))
    300
    """
    return int.from_bytes(buf, "little")


def uncompress(in_file, out_file):
    """ Uncompress contents of in_file and store results in out_file.

    @param str in_file: input file to uncompress
    @param str out_file: output file that will hold the uncompressed results
    @rtype: NoneType
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_size(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(generate_uncompressed(tree, text, size))


# ====================
# Other functions

def improve_tree(tree, freq_dict):
    """ Improve the tree as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to freq_dict.

    @param HuffmanNode tree: Huffman tree rooted at 'tree'
    @param dict(int,int) freq_dict: frequency dictionary
    @rtype: NoneType

    >>> left = HuffmanNode(None, HuffmanNode(99), HuffmanNode(100))
    >>> right = HuffmanNode(None, HuffmanNode(101), \
    HuffmanNode(None, HuffmanNode(97), HuffmanNode(98)))
    >>> tree = HuffmanNode(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # Using the resources from lecture notes, level order.
    acc = []
    for byte in freq_dict:
        acc.append((freq_dict[byte], byte))
    # accoring freq to sort this list.
    acc.sort()
    t = [tree]
    while len(t) != 0:
        t_next = t.pop(0)
        if t_next.is_leaf():
            t_next.symbol = (acc[-1][-1])
            acc.pop()
        if t_next.left:
            t.append(t_next.left)
        if t_next.right:
            t.append(t_next.right)

if __name__ == "__main__":
    import python_ta
    python_ta.check_all(config="huffman_pyta.txt")
    # TODO: Uncomment these when you have implemented all the functions
    # import doctest
    # doctest.testmod()

    import time

    mode = input("Press c to compress or u to uncompress: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress(fname, fname + ".huf")
        print("compressed {} in {} seconds."
              .format(fname, time.time() - start))
    elif mode == "u":
        fname = input("File to uncompress: ")
        start = time.time()
        uncompress(fname, fname + ".orig")
        print("uncompressed {} in {} seconds."
              .format(fname, time.time() - start))
