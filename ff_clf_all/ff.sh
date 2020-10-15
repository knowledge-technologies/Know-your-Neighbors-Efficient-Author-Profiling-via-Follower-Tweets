echo "IN $1"
echo "OUT $2"
python3 /ff_clf_all/src/followers_only.py --persons 10 --idx 10 --input $1 --output $2 
