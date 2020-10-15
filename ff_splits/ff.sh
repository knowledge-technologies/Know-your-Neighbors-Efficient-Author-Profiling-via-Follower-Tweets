echo "IN $1"
echo "OUT $2"
python3 /ff_splits/src/multi_age.py --persons 10 --idx 10 --input $1 --output $2 
