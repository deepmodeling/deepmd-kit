from deepmd.argcheck import gen_doc

def doc_train_input(args):
    doc_str = gen_doc(make_anchor=True)
    print(doc_str)
