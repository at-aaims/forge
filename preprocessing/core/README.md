

CORE raw dataset contains 12,255 compressed tar files. 
After pre-processing, we ended up with a 1.8TB dataset, with 116,149,211 records.

- [File size distribution](#file-size-distribution)
- [Clean up](#clean-up)
- [Rename it (optional)](#rename-it-optional)
- [Data processing](#data-processing)
- [Spark Processing Check](#spark-processing-check)
- [Corrupted tarballs](#corrupted-tarballs)
- [Clean up non-English records](#clean-up-non-english-records)

## File size distribution

```
$ find . -type f -print0 | xargs -0 ls -l | awk '{ n=int(log($5)/log(2)); if (n<10) { n=10; } size[n]++ } END { for (i in size) printf("%d %d\n", 2^i, size[i]) }' | sort -n | awk 'function human(x) { x[1]/=1024; if (x[1]>=1024) { x[2]++; human(x) } } { a[1]=$1; a[2]=0; human(a); printf("%3d%s: %6d\n", a[1],substr("kMGTEPYZ",a[2]+1,1),$2) }'
  1k:   4325
  2k:     23
  4k:     49
  8k:     83
 16k:    123
 32k:    208
 64k:    272
128k:    415
256k:    752
512k:    981
  1M:    990
  2M:    839
  4M:    753
  8M:    597
 16M:    472
 32M:    396
 64M:    335
128M:    276
256M:    210
512M:    100
  1G:     43
  2G:      8
  4G:      3
  8G:      1
 16G:      1
```

## Clean up

A large set of files contains only `manifest.xml` file, with the following content:

```
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
xmlns:rs="http://www.openarchives.org/rs/terms/">
<rs:md capability="resourcedump-manifest"
at="2021-01-20T20:25:31.541226"/> </urlset>
```

we can remove it by
```
find . -size 384c  -delete
```

we now have 8345 files.


## Rename it (optional)

this is rename file with padding zeros

```
ls | awk '/^([0-9]+)\.tar.xz$/ { printf("%s %05d.tar.xz\n", $0, $1) }' | xargs -n2 mv -n
```

## Data processing

Each tar ball get extracted, all json files are combined, using the same base name as their tar ball counter part. For example: `1001.tar.xz` will be combined into a single json file named `1000.json`. 

```python
python merge.py ~/core-raw
```
will have all json files generated `core-outputs`.


## Spark Processing Check

This is exemplified by `sp_count.py` - we read json following using a schema inferred by Spark, and do a count of entries/rows. 

```
spark-submit sp_count.py

Number of rows: 116149211

real    14m40.876s
user    155m51.487s
sys     7m16.507s
```


## Corrupted tarballs

The following files are deemed as corrupted, as not be able to extracted properly.

```
1005.tar.xz   1251.tar.xz   14979.tar.xz  172.tar.xz   2394.tar.xz  3203.tar.xz  340.tar.xz   4843.tar.xz  57.tar.xz    895.tar.xz
10757.tar.xz  12776.tar.xz  153.tar.xz    201.tar.xz   2396.tar.xz  3225.tar.xz  3472.tar.xz  548.tar.xz   636.tar.xz   910.tar.xz
10784.tar.xz  13869.tar.xz  15400.tar.xz  2042.tar.xz  2612.tar.xz  3226.tar.xz  3641.tar.xz  5518.tar.xz  7745.tar.xz  964.tar.xz
10821.tar.xz  14338.tar.xz  1720.tar.xz   2142.tar.xz  292.tar.xz   3376.tar.xz  380.tar.xz   5624.tar.xz  893.tar.xz   983.tar.xz
```

## Clean up non-English records

This is done by use `langdect` on the abstract. Notice that this will drop abstract that is not empty, but no language features. For example, some abstracts has a single number in it - we remove those as well.

After this process, we now have 52,259,422 records. The un-compressed dataset is at 860 GB.



