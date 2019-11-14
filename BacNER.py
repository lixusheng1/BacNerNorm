#-*-coding:utf-8-*-
import os,re
import sys
import nltk,spacy,scispacy
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import tensorflow as tf

def get_chunks(seq):
    """Given a sequence of tags, group entities and their position
    Args:
        seq: ["B-bacteria","I-bacteria","O","O","B-bacteria","O"]equence of labels
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = ["B-bacteria","I-bacteria","O","O","B-bacteria","O"]
        result =[('bacteria', 0, 2), ('bacteria', 4, 5)]
    """
    default = "O"
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = tok.split("-")[0],tok.split("-")[-1]
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def split_sent(sentences):
    i = 0
    while i <= len(sentences) - 2:
        if re.search('^[a-z(>]', sentences[i+1]):
           sentences[i]=sentences[i]+sentences[i+1]
           sentences.remove(sentences[i+1])
           break
        else:
            i += 1
    return sentences

def get_Sentence(article):
    lower_index=[]
    sent_list=nltk.tokenize.sent_tokenize(article)
    for i in range(len(sent_list)):
        if re.search('^[a-z(>]', sent_list[i]):
            lower_index.append(i)
    j=0
    for idnex in lower_index:
        sent_list[idnex-j-1]+=" "+sent_list[idnex-j]
        sent_list.pop(idnex-j)
        j+=1
    return sent_list


def read_file_tokenize(file_path_1,file_path_2):
    if os.path.exists(file_path_2):
        pass
    else:
        os.mkdir(file_path_2)
    if os.path.isdir(file_path_1):
        file_list=os.listdir(file_path_1)
        for file in file_list:
            file_path=os.path.join(file_path_1,file)
            f_path = os.path.join(file_path_2, file)
            fp2 = open(f_path, "w", encoding="utf-8")
            nlp = spacy.load("en_core_sci_sm/en_core_sci_sm/en_core_sci_sm-0.2.4")
            with open(file_path,"r",encoding="utf-8") as fp:
                for line in fp:
                    sent_list=get_Sentence(line)
                    for sent in sent_list:
                        words=nltk.WordPunctTokenizer().tokenize(str(sent))
                        for word in words:
                            fp2.write(word+"\n")
                        fp2.write("\n")

def BacNer(dir_path,save_file_path):
    if os.path.exists(save_file_path):
        pass
    else:
        os.mkdir(save_file_path)
    file_list=os.listdir(dir_path)
    for file in file_list:
        file_path=os.path.join(dir_path,file)
        save_file=os.path.join(save_file_path,file)
        config = Config()
        predict = CoNLLDataset(file_path, config.processing_word, config.max_iter)
        max_sequence_length = max([len(seq[0]) for seq in predict])
        max_word_length = max([len(word[0]) for seq in predict for word in seq[0]])
        model = NERModel(config, max_word_length, max_sequence_length)
        model.build()
        model.restore_session(config.dir_model)
        model.run_predict(predict, save_file)
        tf.reset_default_graph()

def  sentence_ner(dir_path_1):
    if os.path.exists(dir_path_1):
        pass
    else:
        os.mkdir(dir_path_1)
    root=os.getcwd()
    file_list_1=os.listdir(root+"/sentence_tokenize")
    for file in file_list_1:
        file_1=os.path.join(root+"/sentence_tokenize",file)
        file_2=os.path.join(root+"/sentence_ner",file)
        save_file=os.path.join(dir_path_1,file)
        with open(file_1,"r",encoding="utf-8") as fp1:
            sent_list=[]
            sent=[]
            for line in fp1:
                line=line.strip()
                if line:
                    sent.append(line)
                else:
                    if sent:
                        sent_list.append(sent)
                    sent=[]
        with open(file_2,"r",encoding="utf-8") as fp2:
            sent_list_2=[]
            sent_2=[]
            for line in fp2:
                line=line.strip()
                if line:
                    sent_2.append(line)
                else:
                    if sent_2:
                        sent_list_2.append(sent_2)
                    sent_2=[]
        save=open(save_file,"w",encoding="utf-8")
        for i in range(len(sent_list_2)):
            chunks=get_chunks(sent_list_2[i])
            print(" ".join(sent_list[i]))
            if chunks:
                save.write(" ".join(sent_list[i])+"|")
                for entity in chunks:
                    save.write(" ".join(sent_list[i][entity[1]:entity[2]])+"\t"+str(entity[1])+","+str(entity[2])+"|")
                save.write("\n")
        save.close()
        if not os.path.getsize(save_file):
            os.remove(save_file)


if __name__ == '__main__':
    print("please input the path of the file:")
    file_path=sys.stdin.readline().strip()
    root=os.getcwd()
    dir_path_1=root+"/sentence_tokenize"
    dir_path_2=root+"/sentence_ner"
    dir_path_3=root+"/BacNer"
    read_file_tokenize(file_path,dir_path_1)
    BacNer(dir_path_1,dir_path_2)
    sentence_ner(dir_path_3)
    # text="Sharks possess a variety of pathogenic bacteria in their oral cavity that may potentially be transferred into humans during a bite. The aim of the presented study focused on the identification of the bacteria present in the mouths of live blacktip sharks, Carcharhinus limbatus, and the extent that these bacteria possess multi-drug resistance. Swabs were taken from the oral cavity of nineteen live blacktip sharks, which were subsequently released. The average fork length was 146 cm (±11), suggesting the blacktip sharks were mature adults at least 8 years old. All swabs underwent standard microbiological work-up with identification of organisms and reporting of antibiotic susceptibilities using an automated microbiology system. The oral samples revealed an average of 2.72 (±1.4) bacterial isolates per shark. Gram-negative bacteria, making up 61% of all bacterial isolates, were significantly (p<0.001) more common than gram-positive bacteria (39%). The most common organisms were Vibrio spp. (28%), various coagulase-negative Staphylococcus spp. (16%), and Pasteurella spp. (12%). The overall resistance rate was 12% for all antibiotics tested with nearly 43% of bacteria resistant to at least one antibiotic. Multi-drug resistance was seen in 4% of bacteria. No association between shark gender or fork length with bacterial density or antibiotic resistance was observed. Antibiotics with the highest overall susceptibility rates included fluoroquinolones, 3rd generation cephalosporins and sulfamethoxazole/trimethoprim. Recommended empiric antimicrobial therapy for adult blacktip shark bites should encompass either a fluoroquinolone or combination of a 3rd generation cephalosporin plus doxycycline."
    # sents=get_Sentence(text)
    # for s in sents:
    #     print(s)

