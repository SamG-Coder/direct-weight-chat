"""JSON-lines transport around the unchanged offline tensor runtime."""
import argparse
import json
import sys
from pathlib import Path
from chat1gb_autoregressive.engine import Model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--device',default='auto')
    a=p.parse_args()
    compact=(Path(a.model)/'weights.sparse.zip').exists()
    if compact:
        from chat1gb_autoregressive.compact_engine import Model as CompactModel
        m=CompactModel(a.model,device=a.device)
    else:
        m=Model(a.model,device=a.device)
    print(json.dumps(dict(ready=True,device=m.device,compact=compact)),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('action')=='reset':
                m.reset();result=dict(reset=True)
            else:
                result=m.generate(request['prompt'],max_new_tokens=64)
            print(json.dumps(dict(ok=True,result=result)),flush=True)
        except Exception as error:
            print(json.dumps(dict(ok=False,error=str(error))),flush=True)


if __name__=='__main__':main()
