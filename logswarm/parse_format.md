To parse the log file content using the provided log format and regex patterns, we can extract the relevant fields as follows:

### Parsed Log Entries:

1. **Date**: 081109 
   - **Time**: 203615 
   - **Pid**: 148 
   - **Level**: INFO 
   - **Component**: dfs.DataNode$PacketResponder 
   - **Content**: PacketResponder 1 for block blk_38865049064139660 terminating

2. **Date**: 081109 
   - **Time**: 203807 
   - **Pid**: 222 
   - **Level**: INFO 
   - **Component**: dfs.DataNode$PacketResponder 
   - **Content**: PacketResponder 0 for block blk_-6952295868487656571 terminating

3. **Date**: 081109 
   - **Time**: 204005 
   - **Pid**: 35 
   - **Level**: INFO 
   - **Component**: dfs.FSNamesystem 
   - **Content**: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864

4. **Date**: 081109
   - **Time**: 204015
   - **Pid**: 308
   - **Level**: INFO
   - **Component**: dfs.DataNode$PacketResponder
   - **Content**: PacketResponder 2 for block blk_8229193803249955061 terminating

5. **Date**: 081109
   - **Time**: 204106
   - **Pid**: 329
   - **Level**: INFO
   - **Component**: dfs.DataNode$PacketResponder
   - **Content**: PacketResponder 2 for block blk_-6670958622368987959 terminating

6. **Date**: 081109
   - **Time**: 204132
   - **Pid**: 26
   - **Level**: INFO
   - **Component**: dfs.FSNamesystem
   - **Content**: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.43.115:50010 is added to blk_3050920587428079149 size 67108864

7. **Date**: 081109
   - **Time**: 204324
   - **Pid**: 34
   - **Level**: INFO
   - **Component**: dfs.FSNamesystem
   - **Content**: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.203.80:50010 is added to blk_7888946331804732825 size 67108864

8. **Date**: 081109
   - **Time**: 204453
   - **Pid**: 34
   - **Level**: INFO
   - **Component**: dfs.FSNamesystem
   - **Content**: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.11.85:50010 is added to blk_2377150260128098806 size 67108864

9. **Date**: 081109
   - **Time**: 204525
   - **Pid**: 512
   - **Level**: INFO
   - **Component**: dfs.DataNode$PacketResponder
   - **Content**: PacketResponder 2 for block blk_572492839287299681 terminating

10. **Date**: 081109
    - **Time**: 204655
    - **Pid**: 556
    - **Level**: INFO
    - **Component**: dfs.DataNode$PacketResponder
    - **Content**: Received block blk_3587508140051953248 of size 67108864 from /10.251.42.84

11. **Date**: 081109
    - **Time**: 204722
    - **Pid**: 567
    - **Level**: INFO
    - **Component**: dfs.DataNode$PacketResponder
    - **Content**: Received block blk_5402003568334525940 of size 67108864 from /10.251.214.112

12. **Date**: 081109
    - **Time**: 204815
    - **Pid**: 653
    - **Level**: INFO
    - **Component**: dfs.DataNode$DataXceiver
    - **Content**: Receiving block blk_5792489080791696128 src: /10.251.30.6:33145 dest: /10.251.30.6:50010

13. **Date**: 081109
    - **Time**: 204842
    - **Pid**: 663
    - **Level**: INFO
    - **Component**: dfs.DataNode$DataXceiver
    - **Content**: Receiving block blk_1724757848743533110 src: /10.251.111.130:49851 dest: /10.251.111.130:50010

14. **Date**: 081109
    - **Time**: 204908
    - **Pid**: 31
    - **Level**: INFO
    - **Component**: dfs.FSNamesystem
    - **Content**: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.110.8:50010 is added to blk_8015913224713045110 size 67108864

15. **Date**: 081109
    - **Time**: 204925
    - **Pid**: 673
    - **Level**: INFO
    - **Component**: dfs.DataNode$DataXceiver
    - **Content**: Receiving block blk_-5623176793330377570 src: /10.251.75.228:53725 dest: /10.251.75.228:50010

16. **Date**: 081109
    - **Time**: 205035
    - **Pid**: 28
    - **Level**: INFO
    - **Component**: dfs.FSNamesystem
    - **Content**: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000590_0/part-00590. blk_-1727475099218615100

17. **Date**: 081109
    - **Time**: 205056
    - **Pid**: 710
    - **Level**: INFO
    - **Component**: dfs.DataNode$PacketResponder
    - **Content**: PacketResponder 1 for block blk_5017373558217225674 terminating

18. **Date**: 081109
    - **Time**: 205157
    - **Pid**: 752
    - **Level**: INFO
    - **Component**: dfs.DataNode$PacketResponder
    - **Content**: Received block blk_9212264480425680329 of size 67108864 from /10.251.123.1

19. **Date**: 081109
    - **Time**: 205315
    - **Pid**: 29
    - **Level**: INFO
    - **Component**: dfs.FSNamesystem
    - **Content**: BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811092030_0001_m_000742_0/part-00742. blk_-7878121102358435702

20. **Date**: 081109
    - **Time**: 205409
    - **Pid**: 28
    - **Level**: INFO
    - **Component**: dfs.FSNamesystem
    - **Content**: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.251.111.130:50010 is added to blk_4568434182693165548 size 67108864

### Summary
Each log entry has been successfully parsed based on the provided format and regex patterns. If you need additional processing or specific information from these entries, feel free to ask!