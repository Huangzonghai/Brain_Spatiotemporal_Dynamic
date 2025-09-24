import pickle
for j in range(53):
    datas = []
    for i in range(15):
        with open(r"./datas/hbr/a"+str(i+1)+"_3D_total_hbr.pkl", 'rb') as f:
            loaded_data = pickle.load(f)[j]
            nums = 0
            dd = []
            for l in loaded_data:
                for ll in l:
                    dd.append(ll)
                nums = nums+1
                if nums%5==0:
                    datas.append(dd)
                    dd = []
                    nums = 0
            # datas.append(dd)
    for i in range(15):
        with open(r'./datas/hbr/h' + str(i + 1) + '_3D_total_hbr.pkl', 'rb') as f:
            loaded_data = pickle.load(f)[j]
            nums = 0
            dd = []
            for l in loaded_data:
                for ll in l:
                    dd.append(ll)
                nums = nums + 1
                if nums % 5 == 0:
                    datas.append(dd)
                    dd = []
                    nums = 0
            # datas.append(dd)
    print(len(datas))
    with open('./datas/node_hbr/'+str(j+1) + '_hbr_person.pkl', 'wb') as f:
        pickle.dump(datas, f)