import wave
import contextlib
from flacDuration import *
import os
import os.path as osp

file_name_final= 'outputFile.txt'

def list_paths(path):
    directories = [x[1] for x in os.walk(path)]
    non_empty_dirs = [x for x in directories if x] # filter out empty lists
    return [item for subitem in non_empty_dirs for item in subitem] # flatten the list


folders1=list_paths("C:/Users/nicol/Downloads/DNS/train-clean/LibriSpeech/train-clean-100")
for folder1 in folders1:
    folders2=list_paths("C:/Users/nicol/Downloads/DNS/train-clean/LibriSpeech/train-clean-100/"+folder1)
    #print("suc")
    for folder2 in folders2:
        subpath="C:/Users/nicol/Downloads/DNS/train-clean/LibriSpeech/train-clean-100/"+folder1+"/"+folder2+"/";
        mylines = []
        file_name=subpath+folder1+"-"+folder2+'.trans.txt'
        print("suc")

        # Declare an empty list named mylines.
        with open (file_name, 'rt') as myfile: # Open lorem.txt for reading text data.
            for myline in myfile:                # For each line, stored as myline,
                mylines.append(myline)           # add its contents to mylines.

        paths=list()

        s=""
        for line in mylines:
            a=15+len(folder1)+len(folder2)-9
            b=a+1
            if(get_flac_duration(subpath + line[:a] +".flac")<5.2):
                tmp=""
                tmp=tmp+subpath+ line[:a] +".flac|"
                tmp=tmp+tmp
                s=s + tmp + line[b:]
                paths.append(subpath+line[:a] +".flac")
                #print("/home/hartmann/dataset/train-clean-360/1445/138033/"+line[:15] +".flac")

        #print(s)
        file_name= 'temporaryFile.txt'

        with open(file_name, 'w') as f:
            f.write(s)

        with open(file_name, "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                str_to_add=str(get_flac_duration(paths[index]))
                lines[index] = line.strip() +"|"+str_to_add + "\n"

        with open(file_name_final, "a+") as f:
            for line in lines:
                f.write(line)
                #print(line)







