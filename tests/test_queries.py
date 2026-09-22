TEST_QUERIES = [
    {"id":"Q01","type":"direct_match","query":"油温持续升高但是负载变化不大，应该关注什么？","expected_kb":["KB-017"],"should_reject":False},
    {"id":"Q02","type":"paraphrase","query":"负荷没怎么变，但是机器越来越热，应该检查哪里？","expected_kb":["KB-017"],"should_reject":False},
    {"id":"Q03","type":"cooling","query":"油温越来越高，冷却系统需要检查什么？","expected_kb":["KB-017","KB-018","KB-026"],"should_reject":False},
    {"id":"Q04","type":"fan","query":"冷却风扇出现异常会造成什么影响？","expected_kb":["KB-026"],"should_reject":False},
    {"id":"Q05","type":"composite","query":"油温升高同时风扇报警怎么办？","expected_kb":["KB-026","KB-018"],"should_reject":False},
    {"id":"Q06","type":"oil_pump","query":"怀疑散热不好，需要检查油泵吗？","expected_kb":["KB-018"],"should_reject":False},
    {"id":"Q07","type":"temperature_jump","query":"温度突然跳了一下，应该排查什么？","expected_kb":["KB-022"],"should_reject":False},
    {"id":"Q08","type":"sensor","query":"温度采样突然异常，应该检查哪里？","expected_kb":["KB-022"],"should_reject":False},
    {"id":"Q09","type":"overload","query":"设备长期过载会对油温产生什么影响？","expected_kb":["KB-031"],"should_reject":False},
    {"id":"Q10","type":"vibration","query":"设备出现异常振动应该检查什么？","expected_kb":["KB-044"],"should_reject":False},
    {"id":"Q11","type":"out_of_kb_hard","query":"齿轮箱轴向振动频谱怎么分析？","expected_kb":[],"should_reject":True},
    {"id":"Q12","type":"out_of_kb","query":"PLC通讯协议的寄存器地址怎么配置？","expected_kb":[],"should_reject":True}
]
