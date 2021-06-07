# 目录

[toc]

---

## 一、安装

安装包
sbt-1.5.2.msi

安装目录

```
D:\APP\sbt
```

## 二、配置环境变量

安装完之后配置环境变量

1、添加 SBT_HOME
```HTML
SBT_HOME:
D:\APP\sbt
```
2、添加 Path
```html
Path:
%SBT_HOME%\bin
```





## 三、修改 sbt 配置文件


1、修改 `D:\APP\sbt\conf\sbtconfig.txt`

注意

在文件末尾追加
```shell
-Dsbt.log.format=true
-Dsbt.boot.directory=D:/APP/sbt/data/.sbt/boot
-Dsbt.global.base=D:/APP/sbt/data/.sbt
-Dsbt.ivy.home=D:/APP/sbt/data/.ivy2
-Dsbt.repository.config=D:/APP/sbt/conf/repo.properties
-Dsbt.repository.secure=false

# 设置代理
# -Dhttp.proxyHost=10.18.11.11
# -Dhttp.proxyPort=8080
# -Dhttp.proxyUser=xx
# -Dhttp.proxyPassword=xx

# -Dhttps.proxyHost=10.18.1111
# -Dhttps.proxyPort=8080
# -Dhttps.proxyUser=xx
# -Dhttps.proxyPassword=xx
```

2、在 `D:/APP/sbt/conf` 目录下新建 `repo.properties` 文件 repo.properties

设置 repo 的配置，配置为阿里云的镜像
```shell
[repositories]
  local
  aliyun: http://maven.aliyun.com/nexus/content/groups/public/
  typesafe: http://repo.typesafe.com/typesafe/ivy-releases/, [organization]/[module]/(scala_[scalaVersion]/)(sbt_[sbtVersion]/)[revision]/[type]s/[artifact](-[classifier]).[ext], bootOnly
  sonatype-oss-releases
  maven-central
  sonatype-oss-snapshots
```

配置完了，打开 cmd，输入 sbt

一开始报错了
```shell
[error] insecure HTTP request is unsupported 'http://repo1.maven.org/maven2/
...
...
...
...
一堆 Error
```
参考：https://www.cnblogs.com/shuai7boy/p/12568947.html
降低 sbt 的安装版本，版本太高可能国内镜像仓库还没同步
重新下载安装包，从 sbt-1.5.2 换成 sbt-1.3.8