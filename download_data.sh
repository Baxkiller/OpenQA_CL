ROOT="open_domain_data"
DOWNLOAD=$ROOT/"download"

# 如果总参数大于0
while [ "$#" -gt 0 ]; do
    arg=$1
    case $1 in
        # convert "--opt=the value" to --opt "the value".
        # the quotes around the equals sign is to work around a
        # bug in emacs' syntax parsing
        --*'='*) shift; set -- "${arg%%=*}" "${arg#*=}" "$@"; continue;;
        -o|--output) shift; ROOT=$1;;
        -h|--help) usage; exit 0;;
        --) shift; break;;
        -*) usage_fatal "unknown option: '$1'";;
        *) break;; # reached the list of file names
    esac
    shift || usage_fatal "option '${arg}' requires a value"
done


mkdir -p "${ROOT}"
mkdir -p "${DOWNLOAD}"
echo "Saving data in ""$ROOT"

#Wikipedia passages
#主要的文章数据来源
#位置"open_domain_data/pags_w100.tsv"
echo "Downloading Wikipedia passages"
wget -c https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz -P "${DOWNLOAD}"
echo "Decompressing Wikipedia passages"
gzip -d "${DOWNLOAD}/psgs_w100.tsv.gz"
mv "${DOWNLOAD}/psgs_w100.tsv" "${ROOT}/psgs_w100.tsv"

#下载问答对
#这部分数据集包含内容只有{"quesiton":"...","answer":["...",]}
wget -c https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl -P "${DOWNLOAD}"
wget -c https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.train.jsonl -P "${DOWNLOAD}"

#由google预先经过筛选的可用问题，已打乱
wget -c https://dl.fbaipublicfiles.com/FiD/data/dataindex.tar.gz -P "${DOWNLOAD}"
tar xvzf "${DOWNLOAD}/dataindex.tar.gz" -C "${DOWNLOAD}"

#问题相关的passages(内容形式是question id:[passage id])
wget -c https://dl.fbaipublicfiles.com/FiD/data/nq_passages.tar.gz -P "${DOWNLOAD}"
tar xvzf "${DOWNLOAD}/nq_passages.tar.gz" -C "${DOWNLOAD}"

#调用src/preprocess程序，将给定的下载文件和根目录传入
echo "Processing "$ROOT""
python src/preprocess.py $DOWNLOAD $ROOT
rm -r "${DOWNLOAD}"