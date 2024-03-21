import os 
# datasetpath = "/home/yangzho6/c4_parts/downloads/" 
datasetpath = "/data/home/beidic/yang/c4_parts/downloads/" 

files = [
    "c60fe8283b86d4197e18164a07dd5d9b69300493963130a3cd6477bc847766ed ", 
    "aefc3cd300d54ed033387a6a8a8dd445b0d0d2ecb898be4bd15d174f565f41b4", 
    "d1796f4880e8d2c08c340abed80f5263426be1daadbfb1fb4361e13a6a051952", 
    "d44fc77c353b1ae4baf934cc916ff773b5cbefd43c92bfffb35150ea9a76ed04", 
    "ea20f029f84baaaffb94317884396db28952d6ab27c91f5091164df536b77713", 
    "9224fbb43b687e9227cf354fc3258a0833d6a4bd780ab99584864efbbd919ec3", 
    "81ebca015972432d51c7ac7994ed883c2021f5269ef5759b0b7e5d1104dc8f37", 
    "59dbe05ae8639df7a86cecf2e331cbafcf9aee7550087692040ad28c82708a78", 
    "57ff7c96487ae2e0739cc7fb5574861363ae87f19724076238520aa5a75fbce9", 
    "520cce4bc64497bc5daa0bea5d4d4e55b06feeeca5670e4c667daae7a5af8867", 
    "45489bf79bb10a1573057ec1edf3fbf81e1d7ca56afa77ebee4b40cdba47fda9", 
    "36709241ab07df987cd42781626f6359f900d33ce5f6d24027209f31b16c8ae3", 
    "1f793794d3a6d7c82a35556c45c2230e6ef8c8adaca05dd2155f483d284da47f", 
    "1b9a9a06dfafdb7b8e46849487e5411aafe8be74228f479886380e86bf038411", 
    "1535b9418e75602286d29c74147902418f6186c766759e9fb1665c59fa59f9ea"
] 

for i in range(len(files)): 
    os.system("zcat " + datasetpath + files[i] + " > " + datasetpath + "c4_file{}.json".format(i)) 
    print("Done with file {}".format(i)) 
