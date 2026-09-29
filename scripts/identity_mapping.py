"""Panel-local identities; conversions are explicit joins, never offsets."""
def lookup(rows, source, target):
    out={}
    for row in rows:
        key=row[source];value=row[target]
        if key in out or value in out.values():
            raise ValueError('identities must be unique in both directions')
        if not key or not value:
            raise ValueError('identity cannot be empty')
        out[key]=value
    return out

def convert(mapping, key):
    # Unknown identities deliberately raise KeyError.
    return mapping[key]

def align_main(panel, config):
    rows=config.loc[config.primary].to_dict('records')
    mapping=lookup(rows,'submission','figure_id')
    names=sorted(mapping)
    return panel[[mapping[x] for x in names]].to_numpy(float),names

def reverse_estimate(gap,delta,low,high,p,q,state):
    return -gap,-delta,-high,-low,p,q,-state
