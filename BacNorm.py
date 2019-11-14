#-*-coding:utf-8-*-
import os,re
import numpy as np
def get_bacteria_dictionary(file_input):
    bacteria_dict={}
    with open(file_input,"r",encoding="utf-8") as fp:
        for line in fp:
            raw_line=line.strip().split("\t")
            bacteria_id=raw_line[0]
            bacteria_name=[regx_name(name) for name in raw_line[1:]]
            bacteria_dict[bacteria_id]=bacteria_name
    return bacteria_dict

def  regx_name(bacteria_name):
    name=bacteria_name.replace(" ","").lower()
    name=re.sub('\(.*?\)', '', name)
    return name

def bacreria_normlization(file_input,file_output,bacteria_dict):
    input_files=os.listdir(file_input)
    if os.path.exists(file_output):
        pass
    else:
        os.mkdir(file_output)
    for file  in input_files:
        file_path=os.path.join(file_input,file)
        file_write=os.path.join(file_output,file)
        bacteria_name_lsit={}
        sentence_list=[]
        sentence_id=0
        with open(file_path,"r",encoding="utf-8") as fp:
            for line in fp:
                raw_line=line.strip().split("|")[1:-1]
                bacteria_name_lsit[sentence_id]=[regx_name(name.split("\t")[0]) for name in raw_line]
                sentence_list.append(line.strip())
                sentence_id+=1
        for key,value in bacteria_name_lsit.items():
            bac={}
            for name in value:
                bac[name]=""
                for id,id_name in bacteria_dict.items():
                    if re.search(r'spp$',name):
                        if name[:-3] in id_name:
                            bac[name]=id
                            break
                    elif re.search(r'spp\.$',name):
                        if name[:-4] in id_name:
                            bac[name]=id
                            break
                    elif re.search(r"sp$",name):
                        if name[:-2] in id_name:
                            bac[name]=id
                            break
                    elif re.search(r'sp\.$',name):
                        if name[:-3]in id_name:
                            bac[name]=id
                            break
                    elif name in id_name:
                        bac[name] = id
                        break
            bacteria_name_lsit[key]=bac
        # print(bacteria_name_lsit)
        candidate_entity={}
        for key,value in bacteria_name_lsit.items():
            for entity,id in value.items():
                if id:
                    candidate_entity[entity]=id
        for key,value in bacteria_name_lsit.items():
            for entity,id in value.items():
                if not id:
                    index,sim_list=calculate_similarty(entity,candidate_entity.keys())
                    if sim_list[index]>0.9:
                        bacteria_name_lsit[key][entity]=candidate_entity[list(candidate_entity.keys())[index]]
                    else:
                        pass
        # print(bacteria_name_lsit)
        output=open(file_write,"w",encoding="utf-8")
        for sentence_id,entity_id in bacteria_name_lsit.items():
            sent=sentence_list[sentence_id].split("|")[0]
            output.write(sent+"|")
            entity_list=sentence_list[sentence_id].split("|")[1:-1]
            for entity in entity_list:
                entity_name=entity.split("\t")[0]
                entity_cui_id=entity_id[regx_name(entity_name)]
                entity_start_end=entity.split("\t")[-1]
                output.write(entity_name+"\t"+entity_cui_id+"\t"+entity_start_end+"|")
            output.write("\n")




def calculate_similarty(string1,candidate_list):
    import Levenshtein
    sim_list=[]
    for name in candidate_list:
        sim=Levenshtein.jaro_winkler(string1,name)
        sim_list.append(sim)
    print(sim_list)
    index=np.argsort(sim_list)[-1]
    return index,sim_list



if __name__ == '__main__':
    bacteria_dict=get_bacteria_dictionary("bacteria_name.txt")
    bacreria_normlization("BacNer","BacNorm",bacteria_dict)






