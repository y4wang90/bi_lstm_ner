#!/usr/bin/python

import re
import sys

def get_tags(nerTagsMap, key_tag):
    if key_tag in nerTagsMap:
        return nerTagsMap[key_tag]
    else:
        n_nerTags = len(nerTagsMap) + 1
        nerTagsMap[key_tag] = n_nerTags
        return n_nerTags

'''
process the token: get the clean word, remove possessive, plural, etc
'''
def process_token(tok):
    matched = re.search("(\w+)'s$", tok)
    # if matched:
    #     return matched.group(1), True
    # else:
    #     return tok, False
    return (matched.group(1), True) if matched else (tok, False)

def read_xml_train_data(file_name, nerTagsMap, limit=None):
    phrases = []       # training phrase as a list, each row is an list of words
    tags = []          # tag data, each row is an list of index of {0, 1}
    phrases_raw = []

    match_begin_tag=re.compile("^<([\.\w]+)>(.*?)(</[\.\w]+>)?$")
    match_end_tag=re.compile("^(.*?)</([\.\w]+)>$")

    out_count = 0
    with open(file_name) as f:
        for line in f:
            token_s = []        # train phrase a list of words
            tag_s = []          # training label
            line_out = ""
            illegal_flag = 0
            if line[0] == "#": continue
            key_tag = "O"
            #out_sen_buf = []
            key_tokens = line.strip().split(r" ")

            for token in key_tokens:
                #print( "DEBUG: ", token )
                if not token: continue

                # process the token: remove possessive
                (token, bPossessive) = process_token(token)

                matched = match_begin_tag.match(token)
                if matched:
                    #print( "DEBUG:", matched.groups(), matched.group(1), matched.group(2))
                    key_tag = matched.group(1)
                    line_out += matched.group(2).lower() + "\t" + key_tag + "\n"
                    token_s.append(matched.group(2).lower())
                    tag_s.append(get_tags(nerTagsMap, key_tag))

                    #out.write(matched.group(2).lower() + "\t" + key_tag + "\n")
                    if matched.group(3):
                        key_tag = 'O'
                else:
                    matched2 = match_end_tag.match(token)
                    if matched2:
                        if key_tag == matched2.group(2):
                            line_out += matched2.group(1).lower() + "\t" + matched2.group(2)+ "\n"
                            token_s.append(matched2.group(1).lower())
                            tag_s.append(get_tags(nerTagsMap, key_tag))
                            #out.write(matched2.group(1).lower() + "\t" + matched2.group(2)+ "\n")
                        else:
                            #print( line.strip() + "illegal!"
                            illegal_flag = 1

                        key_tag = 'O'

                    else:
                         #out.write(token.lower() + "\t" + key_tag + "\n")
                        line_out += token.lower() + "\t" + key_tag + "\n"
                        token_s.append(token.lower())
                        tag_s.append(0 if key_tag == 'O' else get_tags(nerTagsMap, key_tag))

            if illegal_flag == 0:
                out_count += 1
                if (limit != None and out_count > limit): break
                phrases.append(token_s)
                tags.append(tag_s)
                phrases_raw.append(line)
    return phrases, tags, nerTagsMap, phrases_raw