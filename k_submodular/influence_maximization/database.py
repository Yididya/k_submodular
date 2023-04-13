import os
import sqlite3
import glob
from pathlib import Path
import argparse



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

    def update_db(self, evals_dir, delete=False):
        files = glob.glob(f'{evals_dir}/*.txt')

        print(f'Reading {len(files)}... ')
        items = []

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

                        items.append((key, n_infected, vals[1]))
                except:
                    print('failed adding evaluation to databases')
                    pass
            elif delete:
                # try removing the file --- some could be reading it, so exception may occur
                try:
                    os.remove(f)
                except OSError as e:
                    print('failed deleting a file')
                    pass

        if len(items):
            n = 100 # batch size
            for idx in range(0, len(items), n):
                self.insert_multiple(items[idx: idx+n])

    def insert_item(self, key, value, seed_set):
        try:
            with sqlite3.connect(self.filename) as conn:
                cur = conn.cursor()
                cur.execute(f"insert into t values (   '{key}' ,    {value}, '{seed_set}' )")

        except Exception as e:
            print('Something went wrong inserting item')
            print(e)
            return False

        return True

    def insert_multiple(self, records):
        try:
            with sqlite3.connect(self.filename) as conn:
                cur = conn.cursor()
                cur.executemany("insert into t values (?,?,?);", records)

        except Exception as e:
            print('Something went wrong inserting item')
            print(e)
            return False

        return True






    def fetch_one(self, key):
        query = f'select * from t where id = \'{key}\' '

        try:
            with sqlite3.connect(self.filename) as conn:
                cur = conn.cursor()

                cur.execute(query)
                return cur.fetchone()
        except Exception as e:
            print('Something went wrong fetching values...')
            print(e)
            pass
        return None

    def fetch_n(self, n):

        query = f'select * from t limit {n}'

        try:
            with sqlite3.connect(self.filename) as conn:
                cur = conn.cursor()

                cur.execute(query)
                return cur.fetchall()
        except Exception as e:
            print('Something went wrong fetching values...')
            print(e)
            pass
        return None



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Database aggregator')
    #
    # parser.add_argument('--db', action='store', type=str, required=True)
    # parser.add_argument('--evals-dir', action='store', type=str, required=True)
    #
    # args = parser.parse_args()
    # db_file = args.db
    # evals_dir = args.evals_dir
    #
    # print(f' Evals dir {evals_dir} db file - {db_file}')
    # db = Database(db_file)
    # db.update_db(evals_dir)


    db_file = 'output/evals/evals.db'

    db = Database(db_file)

    new_db = Database('./output/evals/new_db.db')
    new_db.update_db('./output/evals/')
    print('done ')
