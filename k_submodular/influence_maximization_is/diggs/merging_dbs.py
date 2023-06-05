import glob

from k_submodular.database import Database

"""
Script used to merge databases for function evaluations 
"""

if __name__ == '__main__':
    # Merging databases
    root_db = Database(filename='varying_k_results/merged_evals.db')

    db_files = glob.glob('./varying_k_results/db_files/*.db')

    for f in db_files:
        child_db = Database(filename=f)
        values = child_db.fetch_all()
        print(len(values))
        # break
        for i in range(0, len(values), 10000):
            root_db.insert_multiple(values[i: i + 10000])

    # delete duplicates
    root_db.delete_duplicates()

    # writing everything to another file
    final_db = Database(filename='varying_k_results/final_merged_evals.db')

    values = root_db.fetch_all()
    print(len(values))
    for i in range(0, len(values), 10000):
        final_db.insert_multiple(values[i: i + 10000])

    values = final_db.fetch_all()
    print(len(values))
