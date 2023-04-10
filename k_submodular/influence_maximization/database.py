import os
import sqlite3
import glob
from pathlib import Path




class Database:

    def __init__(self, filename='test_new.db'):

        self.filename = filename
        print(self.filename)
        #         self.db = sqlite3.connect(filename)
        #         self.cur = self.db.cursor()

        # create table if not exists
        print('creating database ')
        with sqlite3.connect(self.filename) as conn:
            cur = conn.cursor()
            cur.execute('create table if not exists t (id varchar(300), value real, seed_set text)')

    def update_db(self, evals_dir):
        files = glob.glob(f'{evals_dir}/*.txt')

        for f in files:
            # if not added already
            key = Path(f).stem
            if not self.fetch_one(key):
                # add it

                try:
                    with open(f, 'r') as f_:
                        line = f_.readline()
                        vals = line.strip().split('|')
                        n_infected = float(vals[0].strip())

                        self.insert_item(key, n_infected, seed_set=vals[1])
                except:
                    print('failed adding evaluation to databases')
                    pass
            else:
                # try removing the file --- some could be reading it, so exception may occur
                try:
                    os.remove(f)
                except OSError as e:
                    print('failed deleting a file')
                    pass


    def insert_item(self, key, value, seed_set):
        try:
            with sqlite3.connect(self.filename) as conn:
                cur = conn.cursor()
                cur.execute(f"insert into t values (   '{key}' ,    {value}, '{seed_set}' )")
        except:
            return False

        return True

    def fetch_one(self, key):
        query = f'select * from t where id = \'{key}\' '

        try:
            with sqlite3.connect(self.filename) as conn:
                cur = conn.cursor()

                cur.execute(query)
                return cur.fetchone()
        except:
            pass
        return None