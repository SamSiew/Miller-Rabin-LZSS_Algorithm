"""
Name: Ming Shern, Siew
studentid: 28098552
"""
import sys
from bitarray import bitarray
from bst import BinarySearchTree

def get_all_length(current, length, byte_len, byte_info):
    while current < byte_len and byte_info[current] & 1 == 0:
        new_len = pow(2, length - 1)
        current += 1
        length -= 1
        for i in range(1, length + 1):
            if byte_info[current] & 1 == 1:
                new_len += pow(2, length - 1)
            length -= 1
            current += 1
        length = new_len + 1
    return current, length

def decode_elias(current,length,byte_info):
    new_len = 0
    for i in range(length):
        if byte_info[current] & 1 == 1:
            new_len += pow(2, length - 1)
        length -= 1
        current += 1

    return current,length,new_len


def gen_bits(filename):
    file = open(filename, 'rb+')
    byte_info = bitarray()
    byte_info.fromfile(file)

    byte_len = len(byte_info)

    #header
    current, length = get_all_length(0, 1, byte_len, byte_info)

    #number_of_header
    current, length, new_len = decode_elias(current,length, byte_info)
    new_bst = BinarySearchTree()

    #decode all header
    curr_header = 0
    while curr_header < new_len:
        #decode string(always 7 bit)
        ascii = 0
        for i in range(6,-1,-1):
            if byte_info[current] & 1:
                ascii += pow(2,i)
            current += 1

        #decode elias
        current, length = get_all_length(current, 1, byte_len, byte_info)

        current, length, elias_len = decode_elias(current,length, byte_info)

        #decode codework
        curr_node = new_bst
        for i in range(elias_len):
            if curr_node[byte_info[current] & 1] == None:
                curr_node.add(byte_info[current] & 1)
            curr_node = curr_node[byte_info[current] & 1]
            current += 1
        curr_node.value = ascii
        curr_header += 1

    #decode data
    current, length = get_all_length(current, 1, byte_len, byte_info)
    current, length, new_len = decode_elias(current, length, byte_info)

    data = []
    curr_header = 0
    while curr_header < new_len:
        #decode  bit 0|1
        #decode abc
        if byte_info[current]&1 == 1:
            current += 1
            curr_node = new_bst.root
            while curr_node.value == None:
                curr_node = curr_node[byte_info[current] & 1]
                current += 1
            data.append([1, curr_node.value])
        #decode all num
        else:
            current += 1
            a_data = [0]
            for i in range(2):
                current, length = get_all_length(current, 1, byte_len, byte_info)
                current, length, elias_len = decode_elias(current, length, byte_info)
                a_data.append(elias_len)
            data.append(a_data)
        curr_header += 1


    #todo: data is output now!
    out = "output_decoder_lzss.txt"
    with open(out, 'w+') as outfile:
        for each_data in data:
            if each_data[0] == 1:
                outfile.write(chr(each_data[1]))
            else:
                for _ in range(each_data[2]):
                    pos = outfile.tell()
                    outfile.seek(pos - each_data[1])
                    char = outfile.read(1)
                    outfile.seek(pos)
                    outfile.write(char)


if __name__=="__main__":
    filename = sys.argv[1]
    byte = gen_bits(filename)