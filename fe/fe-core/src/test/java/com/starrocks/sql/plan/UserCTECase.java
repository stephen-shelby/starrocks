package com.starrocks.sql.plan;

import com.starrocks.catalog.OlapTable;
import com.starrocks.catalog.Table;
import com.starrocks.server.GlobalStateMgr;
import com.starrocks.sql.optimizer.statistics.ColumnStatistic;
import com.starrocks.sql.optimizer.statistics.EmptyStatisticStorage;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class UserCTECase extends PlanTestNoneDBBase {
    private static class TestStorage extends EmptyStatisticStorage {
        @Override
        public ColumnStatistic getColumnStatistic(Table table, String column) {
            return new ColumnStatistic(0, 2000000, 0, 8, 2000000);
        }
    }

    @BeforeAll
    public static void beforeClass() throws Exception {
        PlanTestBase.beforeClass();

        PlanTestNoneDBBase.beforeClass();
        String dbName = "db_mock_000";
        starRocksAssert.withDatabase(dbName).useDatabase(dbName);

        GlobalStateMgr globalStateMgr = connectContext.getGlobalStateMgr();
        globalStateMgr.setStatisticStorage(new TestStorage());

//        OlapTable t0 = (OlapTable) globalStateMgr.getLocalMetastore().getDb("db_mock_000").getTable("tbl_mock_001");
//        setTableStatistics(t0, 20000000);
//
//        OlapTable t1 = (OlapTable) globalStateMgr.getLocalMetastore().getDb("db_mock_000").getTable("tbl_mock_065");
//        setTableStatistics(t1, 2000000);
//
//        OlapTable t2 = (OlapTable) globalStateMgr.getLocalMetastore().getDb("db_mock_000").getTable("tbl_mock_125");
//        setTableStatistics(t2, 20000000);
//
//        OlapTable t3 = (OlapTable) globalStateMgr.getLocalMetastore().getDb("db_mock_000").getTable("tbl_mock_136");
//        setTableStatistics(t3, 2000000);
//
//        OlapTable t4 = (OlapTable) globalStateMgr.getLocalMetastore().getDb("db_mock_000").getTable("tbl_mock_154");
//        setTableStatistics(t4, 20000000);
//
//        OlapTable t5 = (OlapTable) globalStateMgr.getLocalMetastore().getDb("db_mock_000").getTable("tbl_mock_156");
//        setTableStatistics(t5, 2000000);
//
//        OlapTable t6 = (OlapTable) globalStateMgr.getLocalMetastore().getDb("db_mock_000").getTable("tbl_mock_188");
//        setTableStatistics(t6, 20000000);
//
//        OlapTable t7 = (OlapTable) globalStateMgr.getLocalMetastore().getDb("db_mock_000").getTable("tbl_mock_190");
//        setTableStatistics(t7, 2000000);
//
//        OlapTable t8 = (OlapTable) globalStateMgr.getLocalMetastore().getDb("db_mock_000").getTable("tbl_mock_192");
//        setTableStatistics(t8, 20000000);


        starRocksAssert.withTable("CREATE TABLE db_mock_000.tbl_mock_001 (\n" +
                "mock_032 largeint(40) NOT NULL ,\n" +
                "mock_057 varchar(65535) ,\n" +
                "mock_033 varchar(65535) ,\n" +
                "mock_056 varchar(65535) ,\n" +
                "mock_049 bigint(20) ,\n" +
                "mock_034 datetime ,\n" +
                "mock_045 datetime ,\n" +
                "mock_048 bigint(20) ,\n" +
                "mock_062 datetime ,\n" +
                "mock_061 datetime ,\n" +
                "mock_004 double ,\n" +
                "mock_035 double ,\n" +
                "mock_058 varchar(65535) ,\n" +
                "mock_036 double ,\n" +
                "mock_064 double ,\n" +
                "mock_060 double ,\n" +
                "mock_063 double ,\n" +
                "mock_059 double ,\n" +
                "mock_046 double ,\n" +
                "mock_003 double ,\n" +
                "mock_005 double ,\n" +
                "mock_037 double ,\n" +
                "mock_055 double ,\n" +
                "mock_053 double ,\n" +
                "mock_042 double ,\n" +
                "mock_040 double ,\n" +
                "mock_051 varchar(65535) ,\n" +
                "mock_038 varchar(65535) ,\n" +
                "mock_054 varchar(65535) ,\n" +
                "mock_041 varchar(65535) ,\n" +
                "mock_002 varchar(65535) ,\n" +
                "mock_052 varchar(65535) ,\n" +
                "mock_039 varchar(65535) ,\n" +
                "id varchar(65535) ,\n" +
                "mock_044 varchar(65535) ,\n" +
                "mock_050 varchar(65535) ,\n" +
                "mock_043 varchar(65535) ,\n" +
                "mock_047 varchar(65535) ,\n" +
                "mock_028 largeint(40) ,\n" +
                "mock_029 largeint(40) ,\n" +
                "mock_030 largeint(40) ,\n" +
                "mock_031 largeint(40) ,\n" +
                "mock_006 bigint(20) ,\n" +
                "mock_007 bigint(20) ,\n" +
                "mock_009 bigint(20) ,\n" +
                "mock_010 bigint(20) ,\n" +
                "mock_011 bigint(20) ,\n" +
                "mock_012 bigint(20) ,\n" +
                "mock_013 bigint(20) ,\n" +
                "mock_014 bigint(20) ,\n" +
                "mock_015 bigint(20) ,\n" +
                "mock_016 bigint(20) ,\n" +
                "mock_008 bigint(20) ,\n" +
                "mock_017 boolean ,\n" +
                "mock_018 boolean ,\n" +
                "mock_020 boolean ,\n" +
                "mock_021 boolean ,\n" +
                "mock_022 boolean ,\n" +
                "mock_023 boolean ,\n" +
                "mock_024 boolean ,\n" +
                "mock_025 boolean ,\n" +
                "mock_026 boolean ,\n" +
                "mock_027 boolean ,\n" +
                "mock_019 boolean \n" +
                ") ENGINE= OLAP \n" +
                "PRIMARY KEY(mock_032)\n" +
                "DISTRIBUTED BY HASH(mock_032) BUCKETS 130 \n" +
                "PROPERTIES (\n" +
                "\"replication_num\" = \"1\"\n" +
                ");");

        starRocksAssert.withTable("CREATE TABLE db_mock_000.tbl_mock_065 (\n" +
                "mock_080 largeint(40) NOT NULL ,\n" +
                "id varchar(65535) ,\n" +
                "mock_083 datetime ,\n" +
                "mock_124 datetime ,\n" +
                "mock_123 datetime ,\n" +
                "mock_114 bigint(20) ,\n" +
                "mock_119 bigint(20) ,\n" +
                "mock_108 varchar(65535) ,\n" +
                "mock_118 varchar(65535) ,\n" +
                "mock_109 bigint(20) ,\n" +
                "mock_115 double ,\n" +
                "mock_113 varchar(65535) ,\n" +
                "mock_111 varchar(65535) ,\n" +
                "mock_112 varchar(65535) ,\n" +
                "mock_120 bigint(20) ,\n" +
                "mock_116 varchar(65535) ,\n" +
                "mock_069 double ,\n" +
                "mock_068 double ,\n" +
                "mock_110 double ,\n" +
                "mock_070 double ,\n" +
                "mock_081 varchar(65535) ,\n" +
                "mock_082 varchar(65535) ,\n" +
                "mock_066 varchar(65535) ,\n" +
                "mock_067 varchar(65535) ,\n" +
                "mock_122 varchar(65535) ,\n" +
                "mock_121 varchar(65535) ,\n" +
                "mock_043 varchar(65535) ,\n" +
                "mock_117 varchar(65535) ,\n" +
                "mock_101 double ,\n" +
                "mock_100 double ,\n" +
                "mock_099 double ,\n" +
                "mock_098 double ,\n" +
                "mock_106 double ,\n" +
                "mock_105 varchar(65535) ,\n" +
                "mock_097 double ,\n" +
                "mock_096 double ,\n" +
                "mock_087 datetime ,\n" +
                "mock_102 varchar(65535) ,\n" +
                "mock_091 bigint(20) ,\n" +
                "mock_094 double ,\n" +
                "mock_088 varchar(65535) ,\n" +
                "mock_085 varchar(65535) ,\n" +
                "mock_084 datetime ,\n" +
                "mock_090 varchar(65535) ,\n" +
                "mock_103 double ,\n" +
                "mock_104 varchar(65535) ,\n" +
                "mock_093 bigint(20) ,\n" +
                "mock_092 bigint(20) ,\n" +
                "mock_086 datetime ,\n" +
                "mock_089 varchar(65535) ,\n" +
                "mock_095 varchar(65535) ,\n" +
                "mock_107 bigint(20) ,\n" +
                "mock_075 largeint(40) ,\n" +
                "mock_076 largeint(40) ,\n" +
                "mock_077 largeint(40) ,\n" +
                "mock_079 largeint(40) ,\n" +
                "mock_074 largeint(40) ,\n" +
                "mock_078 largeint(40) ,\n" +
                "mock_072 largeint(40) ,\n" +
                "mock_071 largeint(40) ,\n" +
                "mock_073 largeint(40) ,\n" +
                "mock_028 largeint(40) ,\n" +
                "mock_017 boolean ,\n" +
                "mock_018 boolean ,\n" +
                "mock_020 boolean ,\n" +
                "mock_021 boolean ,\n" +
                "mock_022 boolean ,\n" +
                "mock_023 boolean ,\n" +
                "mock_024 boolean ,\n" +
                "mock_025 boolean ,\n" +
                "mock_026 boolean ,\n" +
                "mock_027 boolean ,\n" +
                "mock_019 boolean \n" +
                ") ENGINE= OLAP \n" +
                "PRIMARY KEY(mock_080)\n" +
                "DISTRIBUTED BY HASH(mock_080) BUCKETS 366 \n" +
                "PROPERTIES (\n" +
                "\"replication_num\" = \"1\"\n" +
                ");");

        starRocksAssert.withTable("CREATE TABLE db_mock_000.tbl_mock_125 (\n" +
                "mock_126 largeint(40) NOT NULL ,\n" +
                "id varchar(65535) ,\n" +
                "mock_135 varchar(65535) ,\n" +
                "mock_134 varchar(65535) ,\n" +
                "mock_127 varchar(65535) ,\n" +
                "mock_128 varchar(65535) ,\n" +
                "mock_122 varchar(65535) ,\n" +
                "mock_121 varchar(65535) ,\n" +
                "mock_129 varchar(65535) ,\n" +
                "mock_131 varchar(65535) ,\n" +
                "mock_130 varchar(65535) ,\n" +
                "mock_133 varchar(65535) ,\n" +
                "mock_132 varchar(65535) ,\n" +
                "mock_107 bigint(20) ,\n" +
                "mock_077 largeint(40) \n" +
                ") ENGINE= OLAP \n" +
                "PRIMARY KEY(mock_126)\n" +
                "DISTRIBUTED BY HASH(mock_126) BUCKETS 130 \n" +
                "PROPERTIES (\n" +
                "\"replication_num\" = \"1\"\n" +
                ");");

        starRocksAssert.withTable("CREATE TABLE db_mock_000.tbl_mock_136 (\n" +
                "mock_137 largeint(40) NOT NULL ,\n" +
                "id varchar(65535) ,\n" +
                "mock_083 datetime ,\n" +
                "mock_134 varchar(65535) ,\n" +
                "mock_153 varchar(65535) ,\n" +
                "mock_149 varchar(65535) ,\n" +
                "mock_150 varchar(65535) ,\n" +
                "mock_151 varchar(65535) ,\n" +
                "mock_067 varchar(65535) ,\n" +
                "mock_135 varchar(65535) ,\n" +
                "mock_152 varchar(65535) ,\n" +
                "mock_122 varchar(65535) ,\n" +
                "mock_121 varchar(65535) ,\n" +
                "mock_148 varchar(65535) ,\n" +
                "mock_147 varchar(65535) ,\n" +
                "mock_086 datetime ,\n" +
                "mock_093 bigint(20) ,\n" +
                "mock_092 bigint(20) ,\n" +
                "mock_144 bigint(20) ,\n" +
                "mock_139 varchar(65535) ,\n" +
                "mock_138 varchar(65535) ,\n" +
                "mock_146 varchar(65535) ,\n" +
                "mock_145 varchar(65535) ,\n" +
                "mock_141 varchar(65535) ,\n" +
                "mock_140 varchar(65535) ,\n" +
                "mock_143 varchar(65535) ,\n" +
                "mock_142 varchar(65535) ,\n" +
                "mock_107 bigint(20) ,\n" +
                "mock_075 largeint(40) \n" +
                ") ENGINE= OLAP \n" +
                "PRIMARY KEY(mock_137)\n" +
                "DISTRIBUTED BY HASH(mock_137) BUCKETS 130 \n" +
                "PROPERTIES (\n" +
                "\"replication_num\" = \"1\"\n" +
                ");");

        starRocksAssert.withTable("CREATE TABLE db_mock_000.tbl_mock_154 (\n" +
                "mock_155 largeint(40) NOT NULL ,\n" +
                "id varchar(65535) ,\n" +
                "mock_135 varchar(65535) ,\n" +
                "mock_134 varchar(65535) ,\n" +
                "mock_127 varchar(65535) ,\n" +
                "mock_128 varchar(65535) ,\n" +
                "mock_122 varchar(65535) ,\n" +
                "mock_121 varchar(65535) ,\n" +
                "mock_129 varchar(65535) ,\n" +
                "mock_131 varchar(65535) ,\n" +
                "mock_130 varchar(65535) ,\n" +
                "mock_133 varchar(65535) ,\n" +
                "mock_132 varchar(65535) ,\n" +
                "mock_107 bigint(20) ,\n" +
                "mock_030 largeint(40) \n" +
                ") ENGINE= OLAP \n" +
                "PRIMARY KEY(mock_155)\n" +
                "DISTRIBUTED BY HASH(mock_155) BUCKETS 20 \n" +
                "PROPERTIES (\n" +
                "\"replication_num\" = \"1\"\n" +
                ");");

        starRocksAssert.withTable("CREATE TABLE db_mock_000.tbl_mock_156 (\n" +
                "mock_177 largeint(40) NOT NULL ,\n" +
                "mock_006 bigint(20) ,\n" +
                "mock_007 bigint(20) ,\n" +
                "mock_009 bigint(20) ,\n" +
                "mock_010 bigint(20) ,\n" +
                "mock_011 bigint(20) ,\n" +
                "mock_012 bigint(20) ,\n" +
                "mock_013 bigint(20) ,\n" +
                "mock_014 bigint(20) ,\n" +
                "mock_015 bigint(20) ,\n" +
                "mock_016 bigint(20) ,\n" +
                "mock_017 boolean ,\n" +
                "mock_018 boolean ,\n" +
                "mock_020 boolean ,\n" +
                "mock_021 boolean ,\n" +
                "mock_022 boolean ,\n" +
                "mock_023 boolean ,\n" +
                "mock_024 boolean ,\n" +
                "mock_025 boolean ,\n" +
                "mock_026 boolean ,\n" +
                "mock_027 boolean ,\n" +
                "mock_157 datetime ,\n" +
                "mock_158 datetime ,\n" +
                "mock_159 datetime ,\n" +
                "mock_160 datetime ,\n" +
                "mock_161 datetime ,\n" +
                "mock_162 datetime ,\n" +
                "mock_163 datetime ,\n" +
                "mock_164 datetime ,\n" +
                "mock_165 datetime ,\n" +
                "mock_166 datetime ,\n" +
                "mock_167 double ,\n" +
                "mock_168 double ,\n" +
                "mock_169 double ,\n" +
                "mock_170 double ,\n" +
                "mock_171 double ,\n" +
                "mock_172 double ,\n" +
                "mock_173 double ,\n" +
                "mock_174 double ,\n" +
                "mock_175 double ,\n" +
                "mock_176 double ,\n" +
                "mock_178 varchar(65533) ,\n" +
                "mock_179 varchar(65533) ,\n" +
                "mock_180 varchar(65533) ,\n" +
                "mock_181 varchar(65533) ,\n" +
                "mock_182 varchar(65533) ,\n" +
                "mock_183 varchar(65533) ,\n" +
                "mock_184 varchar(65533) ,\n" +
                "mock_185 varchar(65533) ,\n" +
                "mock_186 varchar(65533) ,\n" +
                "mock_187 varchar(65533) \n" +
                ") ENGINE= OLAP \n" +
                "PRIMARY KEY(mock_177)\n" +
                "DISTRIBUTED BY HASH(mock_177) BUCKETS 130 \n" +
                "PROPERTIES (\n" +
                "\"replication_num\" = \"1\"\n" +
                ");");

        starRocksAssert.withTable("CREATE TABLE db_mock_000.tbl_mock_188 (\n" +
                "mock_189 largeint(40) NOT NULL ,\n" +
                "mock_006 bigint(20) ,\n" +
                "mock_007 bigint(20) ,\n" +
                "mock_009 bigint(20) ,\n" +
                "mock_010 bigint(20) ,\n" +
                "mock_011 bigint(20) ,\n" +
                "mock_012 bigint(20) ,\n" +
                "mock_013 bigint(20) ,\n" +
                "mock_014 bigint(20) ,\n" +
                "mock_015 bigint(20) ,\n" +
                "mock_016 bigint(20) ,\n" +
                "mock_017 boolean ,\n" +
                "mock_018 boolean ,\n" +
                "mock_020 boolean ,\n" +
                "mock_021 boolean ,\n" +
                "mock_022 boolean ,\n" +
                "mock_023 boolean ,\n" +
                "mock_024 boolean ,\n" +
                "mock_025 boolean ,\n" +
                "mock_026 boolean ,\n" +
                "mock_027 boolean ,\n" +
                "mock_157 datetime ,\n" +
                "mock_158 datetime ,\n" +
                "mock_159 datetime ,\n" +
                "mock_160 datetime ,\n" +
                "mock_161 datetime ,\n" +
                "mock_162 datetime ,\n" +
                "mock_163 datetime ,\n" +
                "mock_164 datetime ,\n" +
                "mock_165 datetime ,\n" +
                "mock_166 datetime ,\n" +
                "mock_167 double ,\n" +
                "mock_168 double ,\n" +
                "mock_169 double ,\n" +
                "mock_170 double ,\n" +
                "mock_171 double ,\n" +
                "mock_172 double ,\n" +
                "mock_173 double ,\n" +
                "mock_174 double ,\n" +
                "mock_175 double ,\n" +
                "mock_176 double ,\n" +
                "mock_178 varchar(65533) ,\n" +
                "mock_179 varchar(65533) ,\n" +
                "mock_180 varchar(65533) ,\n" +
                "mock_181 varchar(65533) ,\n" +
                "mock_182 varchar(65533) ,\n" +
                "mock_183 varchar(65533) ,\n" +
                "mock_184 varchar(65533) ,\n" +
                "mock_185 varchar(65533) ,\n" +
                "mock_186 varchar(65533) ,\n" +
                "mock_187 varchar(65533) \n" +
                ") ENGINE= OLAP \n" +
                "PRIMARY KEY(mock_189)\n" +
                "DISTRIBUTED BY HASH(mock_189) BUCKETS 130 \n" +
                "PROPERTIES (\n" +
                "\"replication_num\" = \"1\"\n" +
                ");");

        starRocksAssert.withTable("CREATE TABLE db_mock_000.tbl_mock_190 (\n" +
                "mock_191 largeint(40) NOT NULL ,\n" +
                "mock_006 bigint(20) ,\n" +
                "mock_007 bigint(20) ,\n" +
                "mock_009 bigint(20) ,\n" +
                "mock_010 bigint(20) ,\n" +
                "mock_011 bigint(20) ,\n" +
                "mock_012 bigint(20) ,\n" +
                "mock_013 bigint(20) ,\n" +
                "mock_014 bigint(20) ,\n" +
                "mock_015 bigint(20) ,\n" +
                "mock_016 bigint(20) ,\n" +
                "mock_017 boolean ,\n" +
                "mock_018 boolean ,\n" +
                "mock_020 boolean ,\n" +
                "mock_021 boolean ,\n" +
                "mock_022 boolean ,\n" +
                "mock_023 boolean ,\n" +
                "mock_024 boolean ,\n" +
                "mock_025 boolean ,\n" +
                "mock_026 boolean ,\n" +
                "mock_027 boolean ,\n" +
                "mock_157 datetime ,\n" +
                "mock_158 datetime ,\n" +
                "mock_159 datetime ,\n" +
                "mock_160 datetime ,\n" +
                "mock_161 datetime ,\n" +
                "mock_162 datetime ,\n" +
                "mock_163 datetime ,\n" +
                "mock_164 datetime ,\n" +
                "mock_165 datetime ,\n" +
                "mock_166 datetime ,\n" +
                "mock_167 double ,\n" +
                "mock_168 double ,\n" +
                "mock_169 double ,\n" +
                "mock_170 double ,\n" +
                "mock_171 double ,\n" +
                "mock_172 double ,\n" +
                "mock_173 double ,\n" +
                "mock_174 double ,\n" +
                "mock_175 double ,\n" +
                "mock_176 double ,\n" +
                "mock_178 varchar(65533) ,\n" +
                "mock_179 varchar(65533) ,\n" +
                "mock_180 varchar(65533) ,\n" +
                "mock_181 varchar(65533) ,\n" +
                "mock_182 varchar(65533) ,\n" +
                "mock_183 varchar(65533) ,\n" +
                "mock_184 varchar(65533) ,\n" +
                "mock_185 varchar(65533) ,\n" +
                "mock_186 varchar(65533) ,\n" +
                "mock_187 varchar(65533) \n" +
                ") ENGINE= OLAP \n" +
                "PRIMARY KEY(mock_191)\n" +
                "DISTRIBUTED BY HASH(mock_191) BUCKETS 130 \n" +
                "PROPERTIES (\n" +
                "\"replication_num\" = \"1\"\n" +
                ");");

        starRocksAssert.withTable("CREATE TABLE db_mock_000.tbl_mock_192 (\n" +
                "mock_193 largeint(40) NOT NULL ,\n" +
                "mock_006 bigint(20) ,\n" +
                "mock_007 bigint(20) ,\n" +
                "mock_009 bigint(20) ,\n" +
                "mock_010 bigint(20) ,\n" +
                "mock_011 bigint(20) ,\n" +
                "mock_012 bigint(20) ,\n" +
                "mock_013 bigint(20) ,\n" +
                "mock_014 bigint(20) ,\n" +
                "mock_015 bigint(20) ,\n" +
                "mock_016 bigint(20) ,\n" +
                "mock_017 boolean ,\n" +
                "mock_018 boolean ,\n" +
                "mock_020 boolean ,\n" +
                "mock_021 boolean ,\n" +
                "mock_022 boolean ,\n" +
                "mock_023 boolean ,\n" +
                "mock_024 boolean ,\n" +
                "mock_025 boolean ,\n" +
                "mock_026 boolean ,\n" +
                "mock_027 boolean ,\n" +
                "mock_157 datetime ,\n" +
                "mock_158 datetime ,\n" +
                "mock_159 datetime ,\n" +
                "mock_160 datetime ,\n" +
                "mock_161 datetime ,\n" +
                "mock_162 datetime ,\n" +
                "mock_163 datetime ,\n" +
                "mock_164 datetime ,\n" +
                "mock_165 datetime ,\n" +
                "mock_166 datetime ,\n" +
                "mock_167 double ,\n" +
                "mock_168 double ,\n" +
                "mock_169 double ,\n" +
                "mock_170 double ,\n" +
                "mock_171 double ,\n" +
                "mock_172 double ,\n" +
                "mock_173 double ,\n" +
                "mock_174 double ,\n" +
                "mock_175 double ,\n" +
                "mock_176 double ,\n" +
                "mock_178 varchar(65533) ,\n" +
                "mock_179 varchar(65533) ,\n" +
                "mock_180 varchar(65533) ,\n" +
                "mock_181 varchar(65533) ,\n" +
                "mock_182 varchar(65533) ,\n" +
                "mock_183 varchar(65533) ,\n" +
                "mock_184 varchar(65533) ,\n" +
                "mock_185 varchar(65533) ,\n" +
                "mock_186 varchar(65533) ,\n" +
                "mock_187 varchar(65533) \n" +
                ") ENGINE= OLAP \n" +
                "PRIMARY KEY(mock_193)\n" +
                "DISTRIBUTED BY HASH(mock_193) BUCKETS 130 \n" +
                "PROPERTIES (\n" +
                "\"replication_num\" = \"1\"\n" +
                ");\n");


    }

    @Test
    public void testMultiFlatCTE() throws Exception {
        String sql = "WITH tbl_mock_198\n" +
                "(mock_197\n" +
                ") AS (\n" +
                "SELECT  1 AS mock_197)\n" +
                "       ,tbl_mock_206 (mock_028,mock_030,mock_200,mock_201,mock_202,mock_203,mock_204,mock_205,mock_034,id,mock_039,mock_042,mock_022,mock_007,mock_021,mock_006,mock_017,mock_024,mock_018,mock_023,mock_020) AS (\n" +
                "SELECT  tbl_mock_208.mock_028\n" +
                "       ,tbl_mock_208.mock_030\n" +
                "       ,(((tbl_mock_208.mock_055 + tbl_mock_208.mock_042) + (CASE WHEN (tbl_mock_208.mock_046 IS NULL) THEN (CAST(0 AS BIGINT)) ELSE tbl_mock_208.mock_046 END)) * (CAST(365 AS BIGINT))) / (CASE WHEN (((tbl_mock_208.mock_005 IS NULL) OR (tbl_mock_208.mock_005 < (CAST(0 AS BIGINT)))) AND (tbl_mock_208.mock_054 != 'N')) THEN NULL ELSE tbl_mock_208.mock_005 END) AS mock_200\n" +
                "       ,tbl_mock_208.mock_035 * (CAST(0 AS BIGINT))              AS mock_201\n" +
                "       ,xx_hash3_64(tbl_mock_208.mock_058)                       AS mock_202\n" +
                "       ,xx_hash3_64(tbl_mock_208.mock_043)                       AS mock_203\n" +
                "       ,xx_hash3_64(tbl_mock_208.mock_002,tbl_mock_208.mock_039) AS mock_204\n" +
                "       ,xx_hash3_64(tbl_mock_208.mock_039,tbl_mock_208.mock_052) AS mock_205\n" +
                "       ,tbl_mock_208.mock_034\n" +
                "       ,tbl_mock_208.id\n" +
                "       ,tbl_mock_208.mock_039\n" +
                "       ,tbl_mock_208.mock_042\n" +
                "       ,tbl_mock_208.mock_022\n" +
                "       ,tbl_mock_208.mock_007\n" +
                "       ,tbl_mock_208.mock_021\n" +
                "       ,tbl_mock_208.mock_006\n" +
                "       ,tbl_mock_208.mock_017\n" +
                "       ,tbl_mock_208.mock_024\n" +
                "       ,tbl_mock_208.mock_018\n" +
                "       ,tbl_mock_208.mock_023\n" +
                "       ,tbl_mock_208.mock_020\n" +
                "FROM db_mock_000.tbl_mock_001 AS tbl_mock_208) , tbl_mock_001 (mock_028, mock_030, mock_209, mock_210, mock_202, mock_203, mock_204, mock_205, mock_034, id, mock_039, mock_042, mock_021, mock_006, mock_024, mock_020, mock_018, mock_023, mock_007, mock_017, mock_022) AS (\n" +
                "SELECT  tbl_mock_211.mock_028\n" +
                "       ,tbl_mock_211.mock_030\n" +
                "       ,tbl_mock_211.mock_201 * (CAST(3 AS BIGINT)) AS mock_209\n" +
                "       ,tbl_mock_211.mock_200 < (CASE WHEN (((CAST(365000000000 AS BIGINT)) / (CAST(12 AS BIGINT))) > (CAST(0 AS BIGINT))) THEN ((CAST(365000000000 AS BIGINT)) / (CAST(12 AS BIGINT))) ELSE (CAST(9999999999999 AS BIGINT)) END) AS mock_210\n" +
                "       ,tbl_mock_211.mock_202\n" +
                "       ,tbl_mock_211.mock_203\n" +
                "       ,tbl_mock_211.mock_204\n" +
                "       ,tbl_mock_211.mock_205\n" +
                "       ,tbl_mock_211.mock_034\n" +
                "       ,tbl_mock_211.id\n" +
                "       ,tbl_mock_211.mock_039\n" +
                "       ,tbl_mock_211.mock_042\n" +
                "       ,tbl_mock_211.mock_021\n" +
                "       ,tbl_mock_211.mock_006\n" +
                "       ,tbl_mock_211.mock_024\n" +
                "       ,tbl_mock_211.mock_020\n" +
                "       ,tbl_mock_211.mock_018\n" +
                "       ,tbl_mock_211.mock_023\n" +
                "       ,tbl_mock_211.mock_007\n" +
                "       ,tbl_mock_211.mock_017\n" +
                "       ,tbl_mock_211.mock_022\n" +
                "FROM tbl_mock_206 AS tbl_mock_211) , tbl_mock_215 (mock_212, mock_213, mock_214) AS (\n" +
                "SELECT  tbl_mock_216.mock_071 AS mock_212\n" +
                "       ,tbl_mock_217.mock_133 AS mock_213\n" +
                "       ,tbl_mock_218.mock_148 AS mock_214\n" +
                "FROM db_mock_000.tbl_mock_065 AS tbl_mock_216\n" +
                "LEFT OUTER JOIN db_mock_000.tbl_mock_125 AS tbl_mock_217\n" +
                "ON tbl_mock_216.mock_077 = tbl_mock_217.mock_077\n" +
                "LEFT OUTER JOIN db_mock_000.tbl_mock_136 AS tbl_mock_218\n" +
                "ON tbl_mock_216.mock_075 = tbl_mock_218.mock_075) , tbl_mock_239 (mock_219, mock_220, mock_221, mock_222, mock_223, mock_224, mock_225, mock_226, mock_227, mock_228, mock_229, mock_230, mock_231, mock_232, mock_233, mock_234, mock_235, mock_236, mock_237, mock_238, mock_006, mock_007) AS (\n" +
                "SELECT  tbl_mock_240.mock_209 AS mock_219\n" +
                "       ,tbl_mock_243.mock_157 AS mock_220\n" +
                "       ,tbl_mock_244.mock_157 AS mock_221\n" +
                "       ,tbl_mock_241.mock_213 AS mock_222\n" +
                "       ,tbl_mock_242.mock_133 AS mock_223\n" +
                "       ,tbl_mock_241.mock_214 AS mock_224\n" +
                "       ,tbl_mock_240.mock_034 AS mock_225\n" +
                "       ,tbl_mock_240.mock_017 AS mock_226\n" +
                "       ,tbl_mock_240.mock_024 AS mock_227\n" +
                "       ,tbl_mock_240.mock_021 AS mock_228\n" +
                "       ,tbl_mock_240.mock_018 AS mock_229\n" +
                "       ,tbl_mock_240.mock_023 AS mock_230\n" +
                "       ,tbl_mock_240.mock_022 AS mock_231\n" +
                "       ,tbl_mock_240.mock_020 AS mock_232\n" +
                "       ,tbl_mock_240.mock_210 AS mock_233\n" +
                "       ,tbl_mock_240.id       AS mock_234\n" +
                "       ,tbl_mock_240.mock_039 AS mock_235\n" +
                "       ,tbl_mock_240.mock_042 AS mock_236\n" +
                "       ,tbl_mock_240.mock_204 AS mock_237\n" +
                "       ,tbl_mock_240.mock_205 AS mock_238\n" +
                "       ,tbl_mock_240.mock_006\n" +
                "       ,tbl_mock_240.mock_007\n" +
                "FROM tbl_mock_001 AS tbl_mock_240\n" +
                "LEFT OUTER JOIN tbl_mock_215 AS tbl_mock_241\n" +
                "ON tbl_mock_240.mock_028 = tbl_mock_241.mock_212\n" +
                "LEFT OUTER JOIN db_mock_000.tbl_mock_154 AS tbl_mock_242\n" +
                "ON tbl_mock_240.mock_030 = tbl_mock_242.mock_030\n" +
                "LEFT OUTER JOIN db_mock_000.tbl_mock_156 AS tbl_mock_243\n" +
                "ON tbl_mock_240.mock_202 = tbl_mock_243.mock_177\n" +
                "LEFT OUTER JOIN db_mock_000.tbl_mock_188 AS tbl_mock_244\n" +
                "ON tbl_mock_240.mock_203 = tbl_mock_244.mock_189) , tbl_mock_249 (mock_219, mock_224, mock_226, mock_227, mock_228, mock_229, mock_230, mock_231, mock_232, mock_233, mock_234, mock_236, mock_237, mock_238, mock_245, mock_246, mock_247, mock_248, mock_007, mock_006) AS (\n" +
                "SELECT  tbl_mock_250.mock_219\n" +
                "       ,tbl_mock_250.mock_224\n" +
                "       ,tbl_mock_250.mock_226\n" +
                "       ,tbl_mock_250.mock_227\n" +
                "       ,tbl_mock_250.mock_228\n" +
                "       ,tbl_mock_250.mock_229\n" +
                "       ,tbl_mock_250.mock_230\n" +
                "       ,tbl_mock_250.mock_231\n" +
                "       ,tbl_mock_250.mock_232\n" +
                "       ,tbl_mock_250.mock_233\n" +
                "       ,tbl_mock_250.mock_234\n" +
                "       ,tbl_mock_250.mock_236\n" +
                "       ,tbl_mock_250.mock_237\n" +
                "       ,tbl_mock_250.mock_238\n" +
                "       ,CASE tbl_mock_250.mock_222 WHEN tbl_mock_250.mock_223 THEN tbl_mock_250.mock_235 ELSE NULL END       AS mock_245\n" +
                "       ,CASE WHEN (tbl_mock_250.mock_222 != tbl_mock_250.mock_223) THEN tbl_mock_250.mock_235  ELSE NULL END AS mock_246\n" +
                "       ,tbl_mock_250.mock_225 = tbl_mock_250.mock_220                                                        AS mock_247\n" +
                "       ,tbl_mock_250.mock_225 = tbl_mock_250.mock_221                                                        AS mock_248\n" +
                "       ,tbl_mock_250.mock_007\n" +
                "       ,tbl_mock_250.mock_006\n" +
                "FROM tbl_mock_239 AS tbl_mock_250) , tbl_mock_254 (mock_219, mock_224, mock_226, mock_227, mock_228, mock_229, mock_230, mock_231, mock_232, mock_233, mock_234, mock_236, mock_237, mock_238, mock_251, mock_247, mock_248, mock_252, mock_253) AS (\n" +
                "SELECT  tbl_mock_255.mock_219\n" +
                "       ,tbl_mock_255.mock_224\n" +
                "       ,tbl_mock_255.mock_226\n" +
                "       ,tbl_mock_255.mock_227\n" +
                "       ,tbl_mock_255.mock_228\n" +
                "       ,tbl_mock_255.mock_229\n" +
                "       ,tbl_mock_255.mock_230\n" +
                "       ,tbl_mock_255.mock_231\n" +
                "       ,tbl_mock_255.mock_232\n" +
                "       ,tbl_mock_255.mock_233\n" +
                "       ,tbl_mock_255.mock_234\n" +
                "       ,tbl_mock_255.mock_236\n" +
                "       ,tbl_mock_255.mock_237\n" +
                "       ,tbl_mock_255.mock_238\n" +
                "       ,(((((((tbl_mock_255.mock_227 AND tbl_mock_255.mock_228) AND tbl_mock_255.mock_229) AND tbl_mock_255.mock_230) AND tbl_mock_255.mock_226) AND tbl_mock_255.mock_231) AND tbl_mock_255.mock_247) AND tbl_mock_255.mock_248) AND tbl_mock_255.mock_232 AS mock_251\n" +
                "       ,tbl_mock_255.mock_247\n" +
                "       ,tbl_mock_255.mock_248\n" +
                "       ,CASE WHEN (tbl_mock_255.mock_245 IS NULL) THEN NULL  ELSE tbl_mock_255.mock_007 END AS mock_252\n" +
                "       ,CASE WHEN (tbl_mock_255.mock_246 IS NULL) THEN NULL  ELSE tbl_mock_255.mock_006 END AS mock_253\n" +
                "FROM tbl_mock_249 AS tbl_mock_255) , tbl_mock_259 (mock_219, mock_224, mock_226, mock_227, mock_228, mock_229, mock_230, mock_231, mock_232, mock_233, mock_234, mock_237, mock_238, mock_256, mock_257, mock_258, mock_247, mock_248, mock_252, mock_253) AS (\n" +
                "SELECT  tbl_mock_260.mock_219\n" +
                "       ,tbl_mock_260.mock_224\n" +
                "       ,tbl_mock_260.mock_226\n" +
                "       ,tbl_mock_260.mock_227\n" +
                "       ,tbl_mock_260.mock_228\n" +
                "       ,tbl_mock_260.mock_229\n" +
                "       ,tbl_mock_260.mock_230\n" +
                "       ,tbl_mock_260.mock_231\n" +
                "       ,tbl_mock_260.mock_232\n" +
                "       ,tbl_mock_260.mock_233\n" +
                "       ,tbl_mock_260.mock_234\n" +
                "       ,tbl_mock_260.mock_237\n" +
                "       ,tbl_mock_260.mock_238\n" +
                "       ,CASE tbl_mock_260.mock_252 WHEN (CAST(1 AS BIGINT)) THEN tbl_mock_260.mock_236 ELSE (CAST(0 AS BIGINT)) END AS mock_256\n" +
                "       ,CASE tbl_mock_260.mock_253 WHEN (CAST(1 AS BIGINT)) THEN tbl_mock_260.mock_236 ELSE (CAST(0 AS BIGINT)) END AS mock_257\n" +
                "       ,tbl_mock_260.mock_251                                                                                       AS mock_258\n" +
                "       ,tbl_mock_260.mock_247\n" +
                "       ,tbl_mock_260.mock_248\n" +
                "       ,tbl_mock_260.mock_252\n" +
                "       ,tbl_mock_260.mock_253\n" +
                "FROM tbl_mock_254 AS tbl_mock_260) , tbl_mock_264 (mock_261, mock_262, mock_263, mock_237, mock_238) AS (\n" +
                "SELECT  CASE WHEN tbl_mock_265.mock_258 THEN tbl_mock_265.mock_256  ELSE NULL END AS mock_261\n" +
                "       ,tbl_mock_265.mock_258                                                     AS mock_262\n" +
                "       ,CASE WHEN tbl_mock_265.mock_258 THEN tbl_mock_265.mock_257  ELSE NULL END AS mock_263\n" +
                "       ,tbl_mock_265.mock_237\n" +
                "       ,tbl_mock_265.mock_238\n" +
                "FROM tbl_mock_259 AS tbl_mock_265) , tbl_mock_266 (mock_237, mock_261) AS (\n" +
                "SELECT  tbl_mock_267.mock_237\n" +
                "       ,tbl_mock_267.mock_261\n" +
                "FROM tbl_mock_264 AS tbl_mock_267\n" +
                "WHERE (tbl_mock_267.mock_237 IS NOT NULL)\n" +
                "AND tbl_mock_267.mock_262) , tbl_mock_269 (mock_237, mock_268) AS (\n" +
                "SELECT  tbl_mock_267.mock_237\n" +
                "       ,SUM(tbl_mock_267.mock_261) AS mock_268\n" +
                "FROM tbl_mock_266 AS tbl_mock_267\n" +
                "GROUP BY  tbl_mock_267.mock_237)\n" +
                "         ,tbl_mock_270 (mock_238,mock_261,mock_263) AS (\n" +
                "SELECT  tbl_mock_271.mock_238\n" +
                "       ,tbl_mock_271.mock_261\n" +
                "       ,tbl_mock_271.mock_263\n" +
                "FROM tbl_mock_264 AS tbl_mock_271\n" +
                "WHERE (tbl_mock_271.mock_238 IS NOT NULL)\n" +
                "AND (tbl_mock_271.mock_262 OR tbl_mock_271.mock_262)) , tbl_mock_274 (mock_238, mock_272, mock_273) AS (\n" +
                "SELECT  tbl_mock_271.mock_238\n" +
                "       ,SUM(tbl_mock_271.mock_261) AS mock_272\n" +
                "       ,SUM(tbl_mock_271.mock_263) AS mock_273\n" +
                "FROM tbl_mock_270 AS tbl_mock_271\n" +
                "GROUP BY  tbl_mock_271.mock_238)\n" +
                "         ,tbl_mock_277 (mock_275,mock_276) AS (\n" +
                "SELECT  tbl_mock_279.mock_268 AS mock_275\n" +
                "       ,tbl_mock_278.mock_191 AS mock_276\n" +
                "FROM db_mock_000.tbl_mock_190 AS tbl_mock_278\n" +
                "LEFT OUTER JOIN tbl_mock_269 AS tbl_mock_279\n" +
                "ON tbl_mock_278.mock_191 = tbl_mock_279.mock_237) , tbl_mock_283 (mock_280, mock_281, mock_282) AS (\n" +
                "SELECT  tbl_mock_285.mock_272 AS mock_280\n" +
                "       ,tbl_mock_285.mock_273 AS mock_281\n" +
                "       ,tbl_mock_284.mock_193 AS mock_282\n" +
                "FROM db_mock_000.tbl_mock_192 AS tbl_mock_284\n" +
                "LEFT OUTER JOIN tbl_mock_274 AS tbl_mock_285\n" +
                "ON tbl_mock_284.mock_193 = tbl_mock_285.mock_238) , tbl_mock_308 (mock_286, mock_287, mock_288, mock_289, mock_290, mock_291, mock_292, mock_293, mock_294, mock_295, mock_296, mock_297, mock_298, mock_299, mock_300, mock_301, mock_302, mock_303, mock_304, mock_305, mock_306, mock_307) AS (\n" +
                "SELECT  tbl_mock_309.mock_219 AS mock_286\n" +
                "       ,tbl_mock_309.mock_256 AS mock_287\n" +
                "       ,tbl_mock_309.mock_257 AS mock_288\n" +
                "       ,tbl_mock_310.mock_275 AS mock_289\n" +
                "       ,tbl_mock_311.mock_280 AS mock_290\n" +
                "       ,tbl_mock_311.mock_281 AS mock_291\n" +
                "       ,tbl_mock_309.mock_224 AS mock_292\n" +
                "       ,tbl_mock_309.mock_226 AS mock_293\n" +
                "       ,tbl_mock_309.mock_227 AS mock_294\n" +
                "       ,tbl_mock_309.mock_228 AS mock_295\n" +
                "       ,tbl_mock_309.mock_229 AS mock_296\n" +
                "       ,tbl_mock_309.mock_230 AS mock_297\n" +
                "       ,tbl_mock_309.mock_231 AS mock_298\n" +
                "       ,tbl_mock_309.mock_247 AS mock_299\n" +
                "       ,tbl_mock_309.mock_248 AS mock_300\n" +
                "       ,tbl_mock_309.mock_232 AS mock_301\n" +
                "       ,tbl_mock_309.mock_233 AS mock_302\n" +
                "       ,tbl_mock_309.mock_234 AS mock_303\n" +
                "       ,tbl_mock_309.mock_238 AS mock_304\n" +
                "       ,tbl_mock_309.mock_237 AS mock_305\n" +
                "       ,tbl_mock_309.mock_252 AS mock_306\n" +
                "       ,tbl_mock_309.mock_253 AS mock_307\n" +
                "FROM tbl_mock_259 AS tbl_mock_309\n" +
                "LEFT OUTER JOIN tbl_mock_277 AS tbl_mock_310\n" +
                "ON tbl_mock_309.mock_237 = tbl_mock_310.mock_276\n" +
                "LEFT OUTER JOIN tbl_mock_283 AS tbl_mock_311\n" +
                "ON tbl_mock_309.mock_238 = tbl_mock_311.mock_282) , tbl_mock_314 (mock_286, mock_287, mock_288, mock_292, mock_293, mock_294, mock_295, mock_296, mock_297, mock_298, mock_299, mock_300, mock_301, mock_302, mock_303, mock_304, mock_305, mock_306, mock_307, mock_312, mock_313) AS (\n" +
                "SELECT  tbl_mock_315.mock_286\n" +
                "       ,tbl_mock_315.mock_287\n" +
                "       ,tbl_mock_315.mock_288\n" +
                "       ,tbl_mock_315.mock_292\n" +
                "       ,tbl_mock_315.mock_293\n" +
                "       ,tbl_mock_315.mock_294\n" +
                "       ,tbl_mock_315.mock_295\n" +
                "       ,tbl_mock_315.mock_296\n" +
                "       ,tbl_mock_315.mock_297\n" +
                "       ,tbl_mock_315.mock_298\n" +
                "       ,tbl_mock_315.mock_299\n" +
                "       ,tbl_mock_315.mock_300\n" +
                "       ,tbl_mock_315.mock_301\n" +
                "       ,tbl_mock_315.mock_302\n" +
                "       ,tbl_mock_315.mock_303\n" +
                "       ,tbl_mock_315.mock_304\n" +
                "       ,tbl_mock_315.mock_305\n" +
                "       ,tbl_mock_315.mock_306\n" +
                "       ,tbl_mock_315.mock_307\n" +
                "       ,CASE WHEN ((tbl_mock_315.mock_289 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_315.mock_306,CAST(0 AS BIGINT))) = (CAST(1 AS BIGINT)))) THEN tbl_mock_315.mock_290\n" +
                "             WHEN ((tbl_mock_315.mock_289 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_315.mock_306,CAST(0 AS BIGINT))) = (CAST(0 AS BIGINT)))) THEN (CAST(0 AS BIGINT))  ELSE tbl_mock_315.mock_291 END AS mock_312\n" +
                "       ,CASE (CASE WHEN ((tbl_mock_315.mock_289 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_315.mock_306,CAST(0 AS BIGINT))) = (CAST(1 AS BIGINT)))) THEN tbl_mock_315.mock_306 WHEN ((tbl_mock_315.mock_289 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_315.mock_306,CAST(0 AS BIGINT))) != (CAST(1 AS BIGINT)))) THEN (CAST(0 AS BIGINT)) ELSE tbl_mock_315.mock_307 END) WHEN (CAST(1 AS BIGINT)) THEN (CAST(1 AS BIGINT)) ELSE (CAST(0 AS BIGINT)) END AS mock_313\n" +
                "FROM tbl_mock_308 AS tbl_mock_315) , tbl_mock_319 (mock_286, mock_287, mock_288, mock_292, mock_293, mock_294, mock_295, mock_296, mock_297, mock_298, mock_299, mock_300, mock_301, mock_302, mock_303, mock_304, mock_305, mock_306, mock_307, mock_316, mock_317, mock_318) AS (\n" +
                "SELECT  tbl_mock_320.mock_286\n" +
                "       ,tbl_mock_320.mock_287\n" +
                "       ,tbl_mock_320.mock_288\n" +
                "       ,tbl_mock_320.mock_292\n" +
                "       ,tbl_mock_320.mock_293\n" +
                "       ,tbl_mock_320.mock_294\n" +
                "       ,tbl_mock_320.mock_295\n" +
                "       ,tbl_mock_320.mock_296\n" +
                "       ,tbl_mock_320.mock_297\n" +
                "       ,tbl_mock_320.mock_298\n" +
                "       ,tbl_mock_320.mock_299\n" +
                "       ,tbl_mock_320.mock_300\n" +
                "       ,tbl_mock_320.mock_301\n" +
                "       ,tbl_mock_320.mock_302\n" +
                "       ,tbl_mock_320.mock_303\n" +
                "       ,tbl_mock_320.mock_304\n" +
                "       ,tbl_mock_320.mock_305\n" +
                "       ,tbl_mock_320.mock_306\n" +
                "       ,tbl_mock_320.mock_307\n" +
                "       ,CASE WHEN (tbl_mock_320.mock_312 > tbl_mock_320.mock_286) THEN (tbl_mock_320.mock_312 * tbl_mock_320.mock_313)  ELSE (CAST(0 AS BIGINT)) END AS mock_316\n" +
                "       ,CASE tbl_mock_320.mock_313 WHEN (CAST(1 AS BIGINT)) THEN tbl_mock_320.mock_303 ELSE NULL END                                                 AS mock_317\n" +
                "       ,(((((((tbl_mock_320.mock_294 AND tbl_mock_320.mock_295) AND tbl_mock_320.mock_296) AND tbl_mock_320.mock_297) AND tbl_mock_320.mock_293) AND tbl_mock_320.mock_298) AND tbl_mock_320.mock_299) AND tbl_mock_320.mock_300) AND tbl_mock_320.mock_301 AS mock_318\n" +
                "FROM tbl_mock_314 AS tbl_mock_320) , tbl_mock_323 (mock_321, mock_322, mock_304) AS (\n" +
                "SELECT  CASE WHEN tbl_mock_324.mock_318 THEN tbl_mock_324.mock_317  ELSE NULL END AS mock_321\n" +
                "       ,tbl_mock_324.mock_318                                                     AS mock_322\n" +
                "       ,tbl_mock_324.mock_304\n" +
                "FROM tbl_mock_319 AS tbl_mock_324) , tbl_mock_325 (mock_304, mock_321) AS (\n" +
                "SELECT  tbl_mock_326.mock_304\n" +
                "       ,tbl_mock_326.mock_321\n" +
                "FROM tbl_mock_323 AS tbl_mock_326\n" +
                "WHERE (tbl_mock_326.mock_304 IS NOT NULL)\n" +
                "AND tbl_mock_326.mock_322) , tbl_mock_328 (mock_304, mock_327) AS (\n" +
                "SELECT  tbl_mock_326.mock_304\n" +
                "       ,COUNT(tbl_mock_326.mock_321) AS mock_327\n" +
                "FROM tbl_mock_325 AS tbl_mock_326\n" +
                "GROUP BY  tbl_mock_326.mock_304)\n" +
                "         ,tbl_mock_331 (mock_329,mock_330) AS (\n" +
                "SELECT  tbl_mock_333.mock_327 AS mock_329\n" +
                "       ,tbl_mock_332.mock_193 AS mock_330\n" +
                "FROM db_mock_000.tbl_mock_192 AS tbl_mock_332\n" +
                "LEFT OUTER JOIN tbl_mock_328 AS tbl_mock_333\n" +
                "ON tbl_mock_332.mock_193 = tbl_mock_333.mock_304) , tbl_mock_335 (mock_330, mock_334) AS (\n" +
                "SELECT  tbl_mock_336.mock_330\n" +
                "       ,coalesce(tbl_mock_336.mock_329,0) AS mock_334\n" +
                "FROM tbl_mock_331 AS tbl_mock_336) , tbl_mock_358 (mock_337, mock_338, mock_339, mock_340, mock_341, mock_342, mock_343, mock_344, mock_345, mock_346, mock_347, mock_348, mock_349, mock_350, mock_351, mock_352, mock_353, mock_354, mock_355, mock_356, mock_357) AS (\n" +
                "SELECT  tbl_mock_359.mock_316 AS mock_337\n" +
                "       ,tbl_mock_359.mock_287 AS mock_338\n" +
                "       ,tbl_mock_359.mock_288 AS mock_339\n" +
                "       ,tbl_mock_359.mock_286 AS mock_340\n" +
                "       ,tbl_mock_359.mock_292 AS mock_341\n" +
                "       ,tbl_mock_359.mock_293 AS mock_342\n" +
                "       ,tbl_mock_359.mock_294 AS mock_343\n" +
                "       ,tbl_mock_359.mock_295 AS mock_344\n" +
                "       ,tbl_mock_359.mock_296 AS mock_345\n" +
                "       ,tbl_mock_359.mock_297 AS mock_346\n" +
                "       ,tbl_mock_359.mock_298 AS mock_347\n" +
                "       ,tbl_mock_359.mock_299 AS mock_348\n" +
                "       ,tbl_mock_359.mock_300 AS mock_349\n" +
                "       ,tbl_mock_359.mock_301 AS mock_350\n" +
                "       ,tbl_mock_359.mock_302 AS mock_351\n" +
                "       ,tbl_mock_359.mock_303 AS mock_352\n" +
                "       ,tbl_mock_360.mock_334 AS mock_353\n" +
                "       ,tbl_mock_359.mock_305 AS mock_354\n" +
                "       ,tbl_mock_359.mock_304 AS mock_355\n" +
                "       ,tbl_mock_359.mock_306 AS mock_356\n" +
                "       ,tbl_mock_359.mock_307 AS mock_357\n" +
                "FROM tbl_mock_319 AS tbl_mock_359\n" +
                "LEFT OUTER JOIN tbl_mock_335 AS tbl_mock_360\n" +
                "ON tbl_mock_359.mock_304 = tbl_mock_360.mock_330) , tbl_mock_362 (mock_338, mock_339, mock_340, mock_341, mock_342, mock_343, mock_344, mock_345, mock_346, mock_347, mock_348, mock_349, mock_350, mock_351, mock_352, mock_354, mock_355, mock_356, mock_357, mock_361) AS (\n" +
                "SELECT  tbl_mock_363.mock_338\n" +
                "       ,tbl_mock_363.mock_339\n" +
                "       ,tbl_mock_363.mock_340\n" +
                "       ,tbl_mock_363.mock_341\n" +
                "       ,tbl_mock_363.mock_342\n" +
                "       ,tbl_mock_363.mock_343\n" +
                "       ,tbl_mock_363.mock_344\n" +
                "       ,tbl_mock_363.mock_345\n" +
                "       ,tbl_mock_363.mock_346\n" +
                "       ,tbl_mock_363.mock_347\n" +
                "       ,tbl_mock_363.mock_348\n" +
                "       ,tbl_mock_363.mock_349\n" +
                "       ,tbl_mock_363.mock_350\n" +
                "       ,tbl_mock_363.mock_351\n" +
                "       ,tbl_mock_363.mock_352\n" +
                "       ,tbl_mock_363.mock_354\n" +
                "       ,tbl_mock_363.mock_355\n" +
                "       ,tbl_mock_363.mock_356\n" +
                "       ,tbl_mock_363.mock_357\n" +
                "       ,tbl_mock_363.mock_337 / tbl_mock_363.mock_353 AS mock_361\n" +
                "FROM tbl_mock_358 AS tbl_mock_363) , tbl_mock_365 (mock_338, mock_339, mock_340, mock_341, mock_342, mock_343, mock_344, mock_345, mock_346, mock_347, mock_348, mock_349, mock_350, mock_351, mock_352, mock_354, mock_355, mock_356, mock_357, mock_364) AS (\n" +
                "SELECT  tbl_mock_366.mock_338\n" +
                "       ,tbl_mock_366.mock_339\n" +
                "       ,tbl_mock_366.mock_340\n" +
                "       ,tbl_mock_366.mock_341\n" +
                "       ,tbl_mock_366.mock_342\n" +
                "       ,tbl_mock_366.mock_343\n" +
                "       ,tbl_mock_366.mock_344\n" +
                "       ,tbl_mock_366.mock_345\n" +
                "       ,tbl_mock_366.mock_346\n" +
                "       ,tbl_mock_366.mock_347\n" +
                "       ,tbl_mock_366.mock_348\n" +
                "       ,tbl_mock_366.mock_349\n" +
                "       ,tbl_mock_366.mock_350\n" +
                "       ,tbl_mock_366.mock_351\n" +
                "       ,tbl_mock_366.mock_352\n" +
                "       ,tbl_mock_366.mock_354\n" +
                "       ,tbl_mock_366.mock_355\n" +
                "       ,tbl_mock_366.mock_356\n" +
                "       ,tbl_mock_366.mock_357\n" +
                "       ,NOT (tbl_mock_366.mock_361 IS NULL) AS mock_364\n" +
                "FROM tbl_mock_362 AS tbl_mock_366) , tbl_mock_368 (mock_338, mock_339, mock_354, mock_355, mock_367) AS (\n" +
                "SELECT  tbl_mock_369.mock_338\n" +
                "       ,tbl_mock_369.mock_339\n" +
                "       ,tbl_mock_369.mock_354\n" +
                "       ,tbl_mock_369.mock_355\n" +
                "       ,(((((((((tbl_mock_369.mock_343 AND tbl_mock_369.mock_344) AND tbl_mock_369.mock_345) AND tbl_mock_369.mock_346) AND tbl_mock_369.mock_342) AND tbl_mock_369.mock_347) AND tbl_mock_369.mock_348) AND tbl_mock_369.mock_349) AND tbl_mock_369.mock_350) AND tbl_mock_369.mock_364) AND tbl_mock_369.mock_351 AS mock_367\n" +
                "FROM tbl_mock_365 AS tbl_mock_369) , tbl_mock_373 (mock_370, mock_371, mock_372, mock_354, mock_355) AS (\n" +
                "SELECT  CASE WHEN tbl_mock_374.mock_367 THEN tbl_mock_374.mock_338  ELSE NULL END AS mock_370\n" +
                "       ,tbl_mock_374.mock_367                                                     AS mock_371\n" +
                "       ,CASE WHEN tbl_mock_374.mock_367 THEN tbl_mock_374.mock_339  ELSE NULL END AS mock_372\n" +
                "       ,tbl_mock_374.mock_354\n" +
                "       ,tbl_mock_374.mock_355\n" +
                "FROM tbl_mock_368 AS tbl_mock_374) , tbl_mock_375 (mock_354, mock_370) AS (\n" +
                "SELECT  tbl_mock_376.mock_354\n" +
                "       ,tbl_mock_376.mock_370\n" +
                "FROM tbl_mock_373 AS tbl_mock_376\n" +
                "WHERE (tbl_mock_376.mock_354 IS NOT NULL)\n" +
                "AND tbl_mock_376.mock_371) , tbl_mock_378 (mock_354, mock_377) AS (\n" +
                "SELECT  tbl_mock_376.mock_354\n" +
                "       ,SUM(tbl_mock_376.mock_370) AS mock_377\n" +
                "FROM tbl_mock_375 AS tbl_mock_376\n" +
                "GROUP BY  tbl_mock_376.mock_354)\n" +
                "         ,tbl_mock_379 (mock_355,mock_370,mock_372) AS (\n" +
                "SELECT  tbl_mock_380.mock_355\n" +
                "       ,tbl_mock_380.mock_370\n" +
                "       ,tbl_mock_380.mock_372\n" +
                "FROM tbl_mock_373 AS tbl_mock_380\n" +
                "WHERE (tbl_mock_380.mock_355 IS NOT NULL)\n" +
                "AND (tbl_mock_380.mock_371 OR tbl_mock_380.mock_371)) , tbl_mock_383 (mock_355, mock_381, mock_382) AS (\n" +
                "SELECT  tbl_mock_380.mock_355\n" +
                "       ,SUM(tbl_mock_380.mock_370) AS mock_381\n" +
                "       ,SUM(tbl_mock_380.mock_372) AS mock_382\n" +
                "FROM tbl_mock_379 AS tbl_mock_380\n" +
                "GROUP BY  tbl_mock_380.mock_355)\n" +
                "         ,tbl_mock_386 (mock_384,mock_385) AS (\n" +
                "SELECT  tbl_mock_388.mock_377 AS mock_384\n" +
                "       ,tbl_mock_387.mock_191 AS mock_385\n" +
                "FROM db_mock_000.tbl_mock_190 AS tbl_mock_387\n" +
                "LEFT OUTER JOIN tbl_mock_378 AS tbl_mock_388\n" +
                "ON tbl_mock_387.mock_191 = tbl_mock_388.mock_354) , tbl_mock_392 (mock_389, mock_390, mock_391) AS (\n" +
                "SELECT  tbl_mock_394.mock_381 AS mock_389\n" +
                "       ,tbl_mock_394.mock_382 AS mock_390\n" +
                "       ,tbl_mock_393.mock_193 AS mock_391\n" +
                "FROM db_mock_000.tbl_mock_192 AS tbl_mock_393\n" +
                "LEFT OUTER JOIN tbl_mock_383 AS tbl_mock_394\n" +
                "ON tbl_mock_393.mock_193 = tbl_mock_394.mock_355) , tbl_mock_418 (mock_395, mock_396, mock_397, mock_398, mock_399, mock_400, mock_401, mock_402, mock_403, mock_404, mock_405, mock_406, mock_407, mock_408, mock_409, mock_410, mock_411, mock_412, mock_413, mock_414, mock_415, mock_416, mock_417) AS (\n" +
                "SELECT  tbl_mock_419.mock_340 AS mock_395\n" +
                "       ,tbl_mock_419.mock_338 AS mock_396\n" +
                "       ,tbl_mock_419.mock_339 AS mock_397\n" +
                "       ,tbl_mock_420.mock_384 AS mock_398\n" +
                "       ,tbl_mock_421.mock_389 AS mock_399\n" +
                "       ,tbl_mock_421.mock_390 AS mock_400\n" +
                "       ,tbl_mock_419.mock_341 AS mock_401\n" +
                "       ,tbl_mock_419.mock_342 AS mock_402\n" +
                "       ,tbl_mock_419.mock_343 AS mock_403\n" +
                "       ,tbl_mock_419.mock_344 AS mock_404\n" +
                "       ,tbl_mock_419.mock_345 AS mock_405\n" +
                "       ,tbl_mock_419.mock_346 AS mock_406\n" +
                "       ,tbl_mock_419.mock_347 AS mock_407\n" +
                "       ,tbl_mock_419.mock_348 AS mock_408\n" +
                "       ,tbl_mock_419.mock_349 AS mock_409\n" +
                "       ,tbl_mock_419.mock_350 AS mock_410\n" +
                "       ,tbl_mock_419.mock_364 AS mock_411\n" +
                "       ,tbl_mock_419.mock_351 AS mock_412\n" +
                "       ,tbl_mock_419.mock_352 AS mock_413\n" +
                "       ,tbl_mock_419.mock_355 AS mock_414\n" +
                "       ,tbl_mock_419.mock_354 AS mock_415\n" +
                "       ,tbl_mock_419.mock_356 AS mock_416\n" +
                "       ,tbl_mock_419.mock_357 AS mock_417\n" +
                "FROM tbl_mock_365 AS tbl_mock_419\n" +
                "LEFT OUTER JOIN tbl_mock_386 AS tbl_mock_420\n" +
                "ON tbl_mock_419.mock_354 = tbl_mock_420.mock_385\n" +
                "LEFT OUTER JOIN tbl_mock_392 AS tbl_mock_421\n" +
                "ON tbl_mock_419.mock_355 = tbl_mock_421.mock_391) , tbl_mock_424 (mock_395, mock_396, mock_397, mock_401, mock_402, mock_403, mock_404, mock_405, mock_406, mock_407, mock_408, mock_409, mock_410, mock_411, mock_412, mock_413, mock_414, mock_415, mock_416, mock_417, mock_422, mock_423) AS (\n" +
                "SELECT  tbl_mock_425.mock_395\n" +
                "       ,tbl_mock_425.mock_396\n" +
                "       ,tbl_mock_425.mock_397\n" +
                "       ,tbl_mock_425.mock_401\n" +
                "       ,tbl_mock_425.mock_402\n" +
                "       ,tbl_mock_425.mock_403\n" +
                "       ,tbl_mock_425.mock_404\n" +
                "       ,tbl_mock_425.mock_405\n" +
                "       ,tbl_mock_425.mock_406\n" +
                "       ,tbl_mock_425.mock_407\n" +
                "       ,tbl_mock_425.mock_408\n" +
                "       ,tbl_mock_425.mock_409\n" +
                "       ,tbl_mock_425.mock_410\n" +
                "       ,tbl_mock_425.mock_411\n" +
                "       ,tbl_mock_425.mock_412\n" +
                "       ,tbl_mock_425.mock_413\n" +
                "       ,tbl_mock_425.mock_414\n" +
                "       ,tbl_mock_425.mock_415\n" +
                "       ,tbl_mock_425.mock_416\n" +
                "       ,tbl_mock_425.mock_417\n" +
                "       ,CASE WHEN ((tbl_mock_425.mock_398 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_425.mock_416,CAST(0 AS BIGINT))) = (CAST(1 AS BIGINT)))) THEN tbl_mock_425.mock_399\n" +
                "             WHEN ((tbl_mock_425.mock_398 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_425.mock_416,CAST(0 AS BIGINT))) = (CAST(0 AS BIGINT)))) THEN (CAST(0 AS BIGINT))  ELSE tbl_mock_425.mock_400 END AS mock_422\n" +
                "       ,CASE (CASE WHEN ((tbl_mock_425.mock_398 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_425.mock_416,CAST(0 AS BIGINT))) = (CAST(1 AS BIGINT)))) THEN tbl_mock_425.mock_416 WHEN ((tbl_mock_425.mock_398 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_425.mock_416,CAST(0 AS BIGINT))) != (CAST(1 AS BIGINT)))) THEN (CAST(0 AS BIGINT)) ELSE tbl_mock_425.mock_417 END) WHEN (CAST(1 AS BIGINT)) THEN (CAST(1 AS BIGINT)) ELSE (CAST(0 AS BIGINT)) END AS mock_423\n" +
                "FROM tbl_mock_418 AS tbl_mock_425) , tbl_mock_429 (mock_395, mock_396, mock_397, mock_401, mock_402, mock_403, mock_404, mock_405, mock_406, mock_407, mock_408, mock_409, mock_410, mock_411, mock_412, mock_413, mock_414, mock_415, mock_416, mock_417, mock_426, mock_427, mock_428) AS (\n" +
                "SELECT  tbl_mock_430.mock_395\n" +
                "       ,tbl_mock_430.mock_396\n" +
                "       ,tbl_mock_430.mock_397\n" +
                "       ,tbl_mock_430.mock_401\n" +
                "       ,tbl_mock_430.mock_402\n" +
                "       ,tbl_mock_430.mock_403\n" +
                "       ,tbl_mock_430.mock_404\n" +
                "       ,tbl_mock_430.mock_405\n" +
                "       ,tbl_mock_430.mock_406\n" +
                "       ,tbl_mock_430.mock_407\n" +
                "       ,tbl_mock_430.mock_408\n" +
                "       ,tbl_mock_430.mock_409\n" +
                "       ,tbl_mock_430.mock_410\n" +
                "       ,tbl_mock_430.mock_411\n" +
                "       ,tbl_mock_430.mock_412\n" +
                "       ,tbl_mock_430.mock_413\n" +
                "       ,tbl_mock_430.mock_414\n" +
                "       ,tbl_mock_430.mock_415\n" +
                "       ,tbl_mock_430.mock_416\n" +
                "       ,tbl_mock_430.mock_417\n" +
                "       ,CASE WHEN (tbl_mock_430.mock_422 > tbl_mock_430.mock_395) THEN (tbl_mock_430.mock_422 * tbl_mock_430.mock_423)  ELSE (CAST(0 AS BIGINT)) END AS mock_426\n" +
                "       ,CASE tbl_mock_430.mock_423 WHEN (CAST(1 AS BIGINT)) THEN tbl_mock_430.mock_413 ELSE NULL END                                                 AS mock_427\n" +
                "       ,(((((((((tbl_mock_430.mock_403 AND tbl_mock_430.mock_404) AND tbl_mock_430.mock_405) AND tbl_mock_430.mock_406) AND tbl_mock_430.mock_402) AND tbl_mock_430.mock_407) AND tbl_mock_430.mock_408) AND tbl_mock_430.mock_409) AND tbl_mock_430.mock_410) AND tbl_mock_430.mock_411) AND tbl_mock_430.mock_412 AS mock_428\n" +
                "FROM tbl_mock_424 AS tbl_mock_430) , tbl_mock_433 (mock_431, mock_432, mock_414) AS (\n" +
                "SELECT  CASE WHEN tbl_mock_434.mock_428 THEN tbl_mock_434.mock_427  ELSE NULL END AS mock_431\n" +
                "       ,tbl_mock_434.mock_428                                                     AS mock_432\n" +
                "       ,tbl_mock_434.mock_414\n" +
                "FROM tbl_mock_429 AS tbl_mock_434) , tbl_mock_435 (mock_414, mock_431) AS (\n" +
                "SELECT  tbl_mock_436.mock_414\n" +
                "       ,tbl_mock_436.mock_431\n" +
                "FROM tbl_mock_433 AS tbl_mock_436\n" +
                "WHERE (tbl_mock_436.mock_414 IS NOT NULL)\n" +
                "AND tbl_mock_436.mock_432) , tbl_mock_438 (mock_414, mock_437) AS (\n" +
                "SELECT  tbl_mock_436.mock_414\n" +
                "       ,COUNT(tbl_mock_436.mock_431) AS mock_437\n" +
                "FROM tbl_mock_435 AS tbl_mock_436\n" +
                "GROUP BY  tbl_mock_436.mock_414)\n" +
                "         ,tbl_mock_441 (mock_439,mock_440) AS (\n" +
                "SELECT  tbl_mock_443.mock_437 AS mock_439\n" +
                "       ,tbl_mock_442.mock_193 AS mock_440\n" +
                "FROM db_mock_000.tbl_mock_192 AS tbl_mock_442\n" +
                "LEFT OUTER JOIN tbl_mock_438 AS tbl_mock_443\n" +
                "ON tbl_mock_442.mock_193 = tbl_mock_443.mock_414) , tbl_mock_445 (mock_440, mock_444) AS (\n" +
                "SELECT  tbl_mock_446.mock_440\n" +
                "       ,coalesce(tbl_mock_446.mock_439,0) AS mock_444\n" +
                "FROM tbl_mock_441 AS tbl_mock_446) , tbl_mock_469 (mock_447, mock_448, mock_449, mock_450, mock_451, mock_452, mock_453, mock_454, mock_455, mock_456, mock_457, mock_458, mock_459, mock_460, mock_461, mock_462, mock_463, mock_464, mock_465, mock_466, mock_467, mock_468) AS (\n" +
                "SELECT  tbl_mock_470.mock_426 AS mock_447\n" +
                "       ,tbl_mock_470.mock_396 AS mock_448\n" +
                "       ,tbl_mock_470.mock_397 AS mock_449\n" +
                "       ,tbl_mock_470.mock_395 AS mock_450\n" +
                "       ,tbl_mock_470.mock_401 AS mock_451\n" +
                "       ,tbl_mock_470.mock_402 AS mock_452\n" +
                "       ,tbl_mock_470.mock_403 AS mock_453\n" +
                "       ,tbl_mock_470.mock_404 AS mock_454\n" +
                "       ,tbl_mock_470.mock_405 AS mock_455\n" +
                "       ,tbl_mock_470.mock_406 AS mock_456\n" +
                "       ,tbl_mock_470.mock_407 AS mock_457\n" +
                "       ,tbl_mock_470.mock_408 AS mock_458\n" +
                "       ,tbl_mock_470.mock_409 AS mock_459\n" +
                "       ,tbl_mock_470.mock_410 AS mock_460\n" +
                "       ,tbl_mock_470.mock_411 AS mock_461\n" +
                "       ,tbl_mock_470.mock_412 AS mock_462\n" +
                "       ,tbl_mock_470.mock_413 AS mock_463\n" +
                "       ,tbl_mock_471.mock_444 AS mock_464\n" +
                "       ,tbl_mock_470.mock_415 AS mock_465\n" +
                "       ,tbl_mock_470.mock_414 AS mock_466\n" +
                "       ,tbl_mock_470.mock_416 AS mock_467\n" +
                "       ,tbl_mock_470.mock_417 AS mock_468\n" +
                "FROM tbl_mock_429 AS tbl_mock_470\n" +
                "LEFT OUTER JOIN tbl_mock_445 AS tbl_mock_471\n" +
                "ON tbl_mock_470.mock_414 = tbl_mock_471.mock_440) , tbl_mock_473 (mock_448, mock_449, mock_450, mock_451, mock_452, mock_453, mock_454, mock_455, mock_456, mock_457, mock_458, mock_459, mock_460, mock_461, mock_462, mock_463, mock_465, mock_466, mock_467, mock_468, mock_472) AS (\n" +
                "SELECT  tbl_mock_474.mock_448\n" +
                "       ,tbl_mock_474.mock_449\n" +
                "       ,tbl_mock_474.mock_450\n" +
                "       ,tbl_mock_474.mock_451\n" +
                "       ,tbl_mock_474.mock_452\n" +
                "       ,tbl_mock_474.mock_453\n" +
                "       ,tbl_mock_474.mock_454\n" +
                "       ,tbl_mock_474.mock_455\n" +
                "       ,tbl_mock_474.mock_456\n" +
                "       ,tbl_mock_474.mock_457\n" +
                "       ,tbl_mock_474.mock_458\n" +
                "       ,tbl_mock_474.mock_459\n" +
                "       ,tbl_mock_474.mock_460\n" +
                "       ,tbl_mock_474.mock_461\n" +
                "       ,tbl_mock_474.mock_462\n" +
                "       ,tbl_mock_474.mock_463\n" +
                "       ,tbl_mock_474.mock_465\n" +
                "       ,tbl_mock_474.mock_466\n" +
                "       ,tbl_mock_474.mock_467\n" +
                "       ,tbl_mock_474.mock_468\n" +
                "       ,tbl_mock_474.mock_447 / tbl_mock_474.mock_464 AS mock_472\n" +
                "FROM tbl_mock_469 AS tbl_mock_474) , tbl_mock_476 (mock_448, mock_449, mock_450, mock_451, mock_452, mock_453, mock_454, mock_455, mock_456, mock_457, mock_458, mock_459, mock_460, mock_461, mock_462, mock_463, mock_465, mock_466, mock_467, mock_468, mock_475) AS (\n" +
                "SELECT  tbl_mock_477.mock_448\n" +
                "       ,tbl_mock_477.mock_449\n" +
                "       ,tbl_mock_477.mock_450\n" +
                "       ,tbl_mock_477.mock_451\n" +
                "       ,tbl_mock_477.mock_452\n" +
                "       ,tbl_mock_477.mock_453\n" +
                "       ,tbl_mock_477.mock_454\n" +
                "       ,tbl_mock_477.mock_455\n" +
                "       ,tbl_mock_477.mock_456\n" +
                "       ,tbl_mock_477.mock_457\n" +
                "       ,tbl_mock_477.mock_458\n" +
                "       ,tbl_mock_477.mock_459\n" +
                "       ,tbl_mock_477.mock_460\n" +
                "       ,tbl_mock_477.mock_461\n" +
                "       ,tbl_mock_477.mock_462\n" +
                "       ,tbl_mock_477.mock_463\n" +
                "       ,tbl_mock_477.mock_465\n" +
                "       ,tbl_mock_477.mock_466\n" +
                "       ,tbl_mock_477.mock_467\n" +
                "       ,tbl_mock_477.mock_468\n" +
                "       ,tbl_mock_477.mock_472 > (CAST(0 AS BIGINT)) AS mock_475\n" +
                "FROM tbl_mock_473 AS tbl_mock_477) , tbl_mock_479 (mock_448, mock_449, mock_465, mock_466, mock_478) AS (\n" +
                "SELECT  tbl_mock_480.mock_448\n" +
                "       ,tbl_mock_480.mock_449\n" +
                "       ,tbl_mock_480.mock_465\n" +
                "       ,tbl_mock_480.mock_466\n" +
                "       ,((((((((((tbl_mock_480.mock_453 AND tbl_mock_480.mock_454) AND tbl_mock_480.mock_455) AND tbl_mock_480.mock_456) AND tbl_mock_480.mock_452) AND tbl_mock_480.mock_457) AND tbl_mock_480.mock_458) AND tbl_mock_480.mock_459) AND tbl_mock_480.mock_460) AND tbl_mock_480.mock_461) AND tbl_mock_480.mock_462) AND tbl_mock_480.mock_475 AS mock_478\n" +
                "FROM tbl_mock_476 AS tbl_mock_480) , tbl_mock_484 (mock_481, mock_482, mock_483, mock_465, mock_466) AS (\n" +
                "SELECT  CASE WHEN tbl_mock_485.mock_478 THEN tbl_mock_485.mock_448  ELSE NULL END AS mock_481\n" +
                "       ,tbl_mock_485.mock_478                                                     AS mock_482\n" +
                "       ,CASE WHEN tbl_mock_485.mock_478 THEN tbl_mock_485.mock_449  ELSE NULL END AS mock_483\n" +
                "       ,tbl_mock_485.mock_465\n" +
                "       ,tbl_mock_485.mock_466\n" +
                "FROM tbl_mock_479 AS tbl_mock_485) , tbl_mock_486 (mock_465, mock_481) AS (\n" +
                "SELECT  tbl_mock_487.mock_465\n" +
                "       ,tbl_mock_487.mock_481\n" +
                "FROM tbl_mock_484 AS tbl_mock_487\n" +
                "WHERE (tbl_mock_487.mock_465 IS NOT NULL)\n" +
                "AND tbl_mock_487.mock_482) , tbl_mock_489 (mock_465, mock_488) AS (\n" +
                "SELECT  tbl_mock_487.mock_465\n" +
                "       ,SUM(tbl_mock_487.mock_481) AS mock_488\n" +
                "FROM tbl_mock_486 AS tbl_mock_487\n" +
                "GROUP BY  tbl_mock_487.mock_465)\n" +
                "         ,tbl_mock_490 (mock_466,mock_481,mock_483) AS (\n" +
                "SELECT  tbl_mock_491.mock_466\n" +
                "       ,tbl_mock_491.mock_481\n" +
                "       ,tbl_mock_491.mock_483\n" +
                "FROM tbl_mock_484 AS tbl_mock_491\n" +
                "WHERE (tbl_mock_491.mock_466 IS NOT NULL)\n" +
                "AND (tbl_mock_491.mock_482 OR tbl_mock_491.mock_482)) , tbl_mock_494 (mock_466, mock_492, mock_493) AS (\n" +
                "SELECT  tbl_mock_491.mock_466\n" +
                "       ,SUM(tbl_mock_491.mock_481) AS mock_492\n" +
                "       ,SUM(tbl_mock_491.mock_483) AS mock_493\n" +
                "FROM tbl_mock_490 AS tbl_mock_491\n" +
                "GROUP BY  tbl_mock_491.mock_466)\n" +
                "         ,tbl_mock_497 (mock_495,mock_496) AS (\n" +
                "SELECT  tbl_mock_499.mock_488 AS mock_495\n" +
                "       ,tbl_mock_498.mock_191 AS mock_496\n" +
                "FROM db_mock_000.tbl_mock_190 AS tbl_mock_498\n" +
                "LEFT OUTER JOIN tbl_mock_489 AS tbl_mock_499\n" +
                "ON tbl_mock_498.mock_191 = tbl_mock_499.mock_465) , tbl_mock_503 (mock_500, mock_501, mock_502) AS (\n" +
                "SELECT  tbl_mock_505.mock_492 AS mock_500\n" +
                "       ,tbl_mock_505.mock_493 AS mock_501\n" +
                "       ,tbl_mock_504.mock_193 AS mock_502\n" +
                "FROM db_mock_000.tbl_mock_192 AS tbl_mock_504\n" +
                "LEFT OUTER JOIN tbl_mock_494 AS tbl_mock_505\n" +
                "ON tbl_mock_504.mock_193 = tbl_mock_505.mock_466) , tbl_mock_530 (mock_506, mock_507, mock_508, mock_509, mock_510, mock_511, mock_512, mock_513, mock_514, mock_515, mock_516, mock_517, mock_518, mock_519, mock_520, mock_521, mock_522, mock_523, mock_524, mock_525, mock_526, mock_527, mock_528, mock_529) AS (\n" +
                "SELECT  tbl_mock_531.mock_450 AS mock_506\n" +
                "       ,tbl_mock_531.mock_448 AS mock_507\n" +
                "       ,tbl_mock_531.mock_449 AS mock_508\n" +
                "       ,tbl_mock_532.mock_495 AS mock_509\n" +
                "       ,tbl_mock_533.mock_500 AS mock_510\n" +
                "       ,tbl_mock_533.mock_501 AS mock_511\n" +
                "       ,tbl_mock_531.mock_451 AS mock_512\n" +
                "       ,tbl_mock_531.mock_452 AS mock_513\n" +
                "       ,tbl_mock_531.mock_453 AS mock_514\n" +
                "       ,tbl_mock_531.mock_454 AS mock_515\n" +
                "       ,tbl_mock_531.mock_455 AS mock_516\n" +
                "       ,tbl_mock_531.mock_456 AS mock_517\n" +
                "       ,tbl_mock_531.mock_457 AS mock_518\n" +
                "       ,tbl_mock_531.mock_458 AS mock_519\n" +
                "       ,tbl_mock_531.mock_459 AS mock_520\n" +
                "       ,tbl_mock_531.mock_460 AS mock_521\n" +
                "       ,tbl_mock_531.mock_461 AS mock_522\n" +
                "       ,tbl_mock_531.mock_462 AS mock_523\n" +
                "       ,tbl_mock_531.mock_475 AS mock_524\n" +
                "       ,tbl_mock_531.mock_463 AS mock_525\n" +
                "       ,tbl_mock_531.mock_465 AS mock_526\n" +
                "       ,tbl_mock_531.mock_466 AS mock_527\n" +
                "       ,tbl_mock_531.mock_467 AS mock_528\n" +
                "       ,tbl_mock_531.mock_468 AS mock_529\n" +
                "FROM tbl_mock_476 AS tbl_mock_531\n" +
                "LEFT OUTER JOIN tbl_mock_497 AS tbl_mock_532\n" +
                "ON tbl_mock_531.mock_465 = tbl_mock_532.mock_496\n" +
                "LEFT OUTER JOIN tbl_mock_503 AS tbl_mock_533\n" +
                "ON tbl_mock_531.mock_466 = tbl_mock_533.mock_502) , tbl_mock_536 (mock_506, mock_507, mock_508, mock_512, mock_513, mock_514, mock_515, mock_516, mock_517, mock_518, mock_519, mock_520, mock_521, mock_522, mock_523, mock_524, mock_525, mock_526, mock_527, mock_528, mock_529, mock_534, mock_535) AS (\n" +
                "SELECT  tbl_mock_537.mock_506\n" +
                "       ,tbl_mock_537.mock_507\n" +
                "       ,tbl_mock_537.mock_508\n" +
                "       ,tbl_mock_537.mock_512\n" +
                "       ,tbl_mock_537.mock_513\n" +
                "       ,tbl_mock_537.mock_514\n" +
                "       ,tbl_mock_537.mock_515\n" +
                "       ,tbl_mock_537.mock_516\n" +
                "       ,tbl_mock_537.mock_517\n" +
                "       ,tbl_mock_537.mock_518\n" +
                "       ,tbl_mock_537.mock_519\n" +
                "       ,tbl_mock_537.mock_520\n" +
                "       ,tbl_mock_537.mock_521\n" +
                "       ,tbl_mock_537.mock_522\n" +
                "       ,tbl_mock_537.mock_523\n" +
                "       ,tbl_mock_537.mock_524\n" +
                "       ,tbl_mock_537.mock_525\n" +
                "       ,tbl_mock_537.mock_526\n" +
                "       ,tbl_mock_537.mock_527\n" +
                "       ,tbl_mock_537.mock_528\n" +
                "       ,tbl_mock_537.mock_529\n" +
                "       ,CASE WHEN ((tbl_mock_537.mock_509 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_537.mock_528,CAST(0 AS BIGINT))) = (CAST(1 AS BIGINT)))) THEN tbl_mock_537.mock_510\n" +
                "             WHEN ((tbl_mock_537.mock_509 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_537.mock_528,CAST(0 AS BIGINT))) = (CAST(0 AS BIGINT)))) THEN (CAST(0 AS BIGINT))  ELSE tbl_mock_537.mock_511 END AS mock_534\n" +
                "       ,CASE (CASE WHEN ((tbl_mock_537.mock_509 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_537.mock_528,CAST(0 AS BIGINT))) = (CAST(1 AS BIGINT)))) THEN tbl_mock_537.mock_528 WHEN ((tbl_mock_537.mock_509 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_537.mock_528,CAST(0 AS BIGINT))) != (CAST(1 AS BIGINT)))) THEN (CAST(0 AS BIGINT)) ELSE tbl_mock_537.mock_529 END) WHEN (CAST(1 AS BIGINT)) THEN (CAST(1 AS BIGINT)) ELSE (CAST(0 AS BIGINT)) END AS mock_535\n" +
                "FROM tbl_mock_530 AS tbl_mock_537) , tbl_mock_539 (mock_506, mock_507, mock_508, mock_512, mock_513, mock_514, mock_515, mock_516, mock_517, mock_518, mock_519, mock_520, mock_521, mock_522, mock_523, mock_524, mock_525, mock_526, mock_527, mock_528, mock_529, mock_538) AS (\n" +
                "SELECT  tbl_mock_540.mock_506\n" +
                "       ,tbl_mock_540.mock_507\n" +
                "       ,tbl_mock_540.mock_508\n" +
                "       ,tbl_mock_540.mock_512\n" +
                "       ,tbl_mock_540.mock_513\n" +
                "       ,tbl_mock_540.mock_514\n" +
                "       ,tbl_mock_540.mock_515\n" +
                "       ,tbl_mock_540.mock_516\n" +
                "       ,tbl_mock_540.mock_517\n" +
                "       ,tbl_mock_540.mock_518\n" +
                "       ,tbl_mock_540.mock_519\n" +
                "       ,tbl_mock_540.mock_520\n" +
                "       ,tbl_mock_540.mock_521\n" +
                "       ,tbl_mock_540.mock_522\n" +
                "       ,tbl_mock_540.mock_523\n" +
                "       ,tbl_mock_540.mock_524\n" +
                "       ,tbl_mock_540.mock_525\n" +
                "       ,tbl_mock_540.mock_526\n" +
                "       ,tbl_mock_540.mock_527\n" +
                "       ,tbl_mock_540.mock_528\n" +
                "       ,tbl_mock_540.mock_529\n" +
                "       ,CASE WHEN (tbl_mock_540.mock_534 > tbl_mock_540.mock_506) THEN (tbl_mock_540.mock_534 * tbl_mock_540.mock_535)  ELSE (CAST(0 AS BIGINT)) END AS mock_538\n" +
                "FROM tbl_mock_536 AS tbl_mock_540) , tbl_mock_542 (mock_506, mock_507, mock_508, mock_512, mock_513, mock_514, mock_515, mock_516, mock_517, mock_518, mock_519, mock_520, mock_521, mock_522, mock_523, mock_524, mock_525, mock_526, mock_527, mock_528, mock_529, mock_541) AS (\n" +
                "SELECT  tbl_mock_543.mock_506\n" +
                "       ,tbl_mock_543.mock_507\n" +
                "       ,tbl_mock_543.mock_508\n" +
                "       ,tbl_mock_543.mock_512\n" +
                "       ,tbl_mock_543.mock_513\n" +
                "       ,tbl_mock_543.mock_514\n" +
                "       ,tbl_mock_543.mock_515\n" +
                "       ,tbl_mock_543.mock_516\n" +
                "       ,tbl_mock_543.mock_517\n" +
                "       ,tbl_mock_543.mock_518\n" +
                "       ,tbl_mock_543.mock_519\n" +
                "       ,tbl_mock_543.mock_520\n" +
                "       ,tbl_mock_543.mock_521\n" +
                "       ,tbl_mock_543.mock_522\n" +
                "       ,tbl_mock_543.mock_523\n" +
                "       ,tbl_mock_543.mock_524\n" +
                "       ,tbl_mock_543.mock_525\n" +
                "       ,tbl_mock_543.mock_526\n" +
                "       ,tbl_mock_543.mock_527\n" +
                "       ,tbl_mock_543.mock_528\n" +
                "       ,tbl_mock_543.mock_529\n" +
                "       ,tbl_mock_543.mock_538 > (CAST(0 AS BIGINT)) AS mock_541\n" +
                "FROM tbl_mock_539 AS tbl_mock_543) , tbl_mock_545 (mock_507, mock_508, mock_526, mock_527, mock_544) AS (\n" +
                "SELECT  tbl_mock_546.mock_507\n" +
                "       ,tbl_mock_546.mock_508\n" +
                "       ,tbl_mock_546.mock_526\n" +
                "       ,tbl_mock_546.mock_527\n" +
                "       ,(((((((((((tbl_mock_546.mock_514 AND tbl_mock_546.mock_515) AND tbl_mock_546.mock_516) AND tbl_mock_546.mock_517) AND tbl_mock_546.mock_513) AND tbl_mock_546.mock_518) AND tbl_mock_546.mock_519) AND tbl_mock_546.mock_520) AND tbl_mock_546.mock_521) AND tbl_mock_546.mock_522) AND tbl_mock_546.mock_523) AND tbl_mock_546.mock_524) AND tbl_mock_546.mock_541 AS mock_544\n" +
                "FROM tbl_mock_542 AS tbl_mock_546) , tbl_mock_550 (mock_547, mock_548, mock_549, mock_526, mock_527) AS (\n" +
                "SELECT  CASE WHEN tbl_mock_551.mock_544 THEN tbl_mock_551.mock_507  ELSE NULL END AS mock_547\n" +
                "       ,tbl_mock_551.mock_544                                                     AS mock_548\n" +
                "       ,CASE WHEN tbl_mock_551.mock_544 THEN tbl_mock_551.mock_508  ELSE NULL END AS mock_549\n" +
                "       ,tbl_mock_551.mock_526\n" +
                "       ,tbl_mock_551.mock_527\n" +
                "FROM tbl_mock_545 AS tbl_mock_551) , tbl_mock_552 (mock_526, mock_547) AS (\n" +
                "SELECT  tbl_mock_553.mock_526\n" +
                "       ,tbl_mock_553.mock_547\n" +
                "FROM tbl_mock_550 AS tbl_mock_553\n" +
                "WHERE (tbl_mock_553.mock_526 IS NOT NULL)\n" +
                "AND tbl_mock_553.mock_548) , tbl_mock_555 (mock_526, mock_554) AS (\n" +
                "SELECT  tbl_mock_553.mock_526\n" +
                "       ,SUM(tbl_mock_553.mock_547) AS mock_554\n" +
                "FROM tbl_mock_552 AS tbl_mock_553\n" +
                "GROUP BY  tbl_mock_553.mock_526)\n" +
                "         ,tbl_mock_556 (mock_527,mock_547,mock_549) AS (\n" +
                "SELECT  tbl_mock_557.mock_527\n" +
                "       ,tbl_mock_557.mock_547\n" +
                "       ,tbl_mock_557.mock_549\n" +
                "FROM tbl_mock_550 AS tbl_mock_557\n" +
                "WHERE (tbl_mock_557.mock_527 IS NOT NULL)\n" +
                "AND (tbl_mock_557.mock_548 OR tbl_mock_557.mock_548)) , tbl_mock_560 (mock_527, mock_558, mock_559) AS (\n" +
                "SELECT  tbl_mock_557.mock_527\n" +
                "       ,SUM(tbl_mock_557.mock_547) AS mock_558\n" +
                "       ,SUM(tbl_mock_557.mock_549) AS mock_559\n" +
                "FROM tbl_mock_556 AS tbl_mock_557\n" +
                "GROUP BY  tbl_mock_557.mock_527)\n" +
                "         ,tbl_mock_563 (mock_561,mock_562) AS (\n" +
                "SELECT  tbl_mock_565.mock_554 AS mock_561\n" +
                "       ,tbl_mock_564.mock_191 AS mock_562\n" +
                "FROM db_mock_000.tbl_mock_190 AS tbl_mock_564\n" +
                "LEFT OUTER JOIN tbl_mock_555 AS tbl_mock_565\n" +
                "ON tbl_mock_564.mock_191 = tbl_mock_565.mock_526) , tbl_mock_569 (mock_566, mock_567, mock_568) AS (\n" +
                "SELECT  tbl_mock_571.mock_558 AS mock_566\n" +
                "       ,tbl_mock_571.mock_559 AS mock_567\n" +
                "       ,tbl_mock_570.mock_193 AS mock_568\n" +
                "FROM db_mock_000.tbl_mock_192 AS tbl_mock_570\n" +
                "LEFT OUTER JOIN tbl_mock_560 AS tbl_mock_571\n" +
                "ON tbl_mock_570.mock_193 = tbl_mock_571.mock_527) , tbl_mock_594 (mock_572, mock_573, mock_574, mock_575, mock_576, mock_577, mock_578, mock_579, mock_580, mock_581, mock_582, mock_583, mock_584, mock_585, mock_586, mock_587, mock_588, mock_589, mock_590, mock_591, mock_592, mock_593) AS (\n" +
                "SELECT  tbl_mock_595.mock_506 AS mock_572\n" +
                "       ,tbl_mock_596.mock_561 AS mock_573\n" +
                "       ,tbl_mock_597.mock_566 AS mock_574\n" +
                "       ,tbl_mock_597.mock_567 AS mock_575\n" +
                "       ,tbl_mock_595.mock_512 AS mock_576\n" +
                "       ,tbl_mock_595.mock_513 AS mock_577\n" +
                "       ,tbl_mock_595.mock_514 AS mock_578\n" +
                "       ,tbl_mock_595.mock_515 AS mock_579\n" +
                "       ,tbl_mock_595.mock_516 AS mock_580\n" +
                "       ,tbl_mock_595.mock_517 AS mock_581\n" +
                "       ,tbl_mock_595.mock_518 AS mock_582\n" +
                "       ,tbl_mock_595.mock_519 AS mock_583\n" +
                "       ,tbl_mock_595.mock_520 AS mock_584\n" +
                "       ,tbl_mock_595.mock_521 AS mock_585\n" +
                "       ,tbl_mock_595.mock_522 AS mock_586\n" +
                "       ,tbl_mock_595.mock_523 AS mock_587\n" +
                "       ,tbl_mock_595.mock_524 AS mock_588\n" +
                "       ,tbl_mock_595.mock_541 AS mock_589\n" +
                "       ,tbl_mock_595.mock_525 AS mock_590\n" +
                "       ,tbl_mock_595.mock_527 AS mock_591\n" +
                "       ,tbl_mock_595.mock_528 AS mock_592\n" +
                "       ,tbl_mock_595.mock_529 AS mock_593\n" +
                "FROM tbl_mock_542 AS tbl_mock_595\n" +
                "LEFT OUTER JOIN tbl_mock_563 AS tbl_mock_596\n" +
                "ON tbl_mock_595.mock_526 = tbl_mock_596.mock_562\n" +
                "LEFT OUTER JOIN tbl_mock_569 AS tbl_mock_597\n" +
                "ON tbl_mock_595.mock_527 = tbl_mock_597.mock_568) , tbl_mock_600 (mock_572, mock_576, mock_577, mock_578, mock_579, mock_580, mock_581, mock_582, mock_583, mock_584, mock_585, mock_586, mock_587, mock_588, mock_589, mock_590, mock_591, mock_598, mock_599) AS (\n" +
                "SELECT  tbl_mock_601.mock_572\n" +
                "       ,tbl_mock_601.mock_576\n" +
                "       ,tbl_mock_601.mock_577\n" +
                "       ,tbl_mock_601.mock_578\n" +
                "       ,tbl_mock_601.mock_579\n" +
                "       ,tbl_mock_601.mock_580\n" +
                "       ,tbl_mock_601.mock_581\n" +
                "       ,tbl_mock_601.mock_582\n" +
                "       ,tbl_mock_601.mock_583\n" +
                "       ,tbl_mock_601.mock_584\n" +
                "       ,tbl_mock_601.mock_585\n" +
                "       ,tbl_mock_601.mock_586\n" +
                "       ,tbl_mock_601.mock_587\n" +
                "       ,tbl_mock_601.mock_588\n" +
                "       ,tbl_mock_601.mock_589\n" +
                "       ,tbl_mock_601.mock_590\n" +
                "       ,tbl_mock_601.mock_591\n" +
                "       ,CASE WHEN ((tbl_mock_601.mock_573 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_601.mock_592,CAST(0 AS BIGINT))) = (CAST(1 AS BIGINT)))) THEN tbl_mock_601.mock_574\n" +
                "             WHEN ((tbl_mock_601.mock_573 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_601.mock_592,CAST(0 AS BIGINT))) = (CAST(0 AS BIGINT)))) THEN (CAST(0 AS BIGINT))  ELSE tbl_mock_601.mock_575 END AS mock_598\n" +
                "       ,CASE (CASE WHEN ((tbl_mock_601.mock_573 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_601.mock_592,CAST(0 AS BIGINT))) = (CAST(1 AS BIGINT)))) THEN tbl_mock_601.mock_592 WHEN ((tbl_mock_601.mock_573 > (CAST(0 AS BIGINT))) AND ((coalesce(tbl_mock_601.mock_592,CAST(0 AS BIGINT))) != (CAST(1 AS BIGINT)))) THEN (CAST(0 AS BIGINT)) ELSE tbl_mock_601.mock_593 END) WHEN (CAST(1 AS BIGINT)) THEN (CAST(1 AS BIGINT)) ELSE (CAST(0 AS BIGINT)) END AS mock_599\n" +
                "FROM tbl_mock_594 AS tbl_mock_601) , tbl_mock_605 (mock_576, mock_577, mock_578, mock_579, mock_580, mock_581, mock_582, mock_583, mock_584, mock_585, mock_586, mock_587, mock_588, mock_589, mock_591, mock_602, mock_603, mock_604) AS (\n" +
                "SELECT  tbl_mock_606.mock_576\n" +
                "       ,tbl_mock_606.mock_577\n" +
                "       ,tbl_mock_606.mock_578\n" +
                "       ,tbl_mock_606.mock_579\n" +
                "       ,tbl_mock_606.mock_580\n" +
                "       ,tbl_mock_606.mock_581\n" +
                "       ,tbl_mock_606.mock_582\n" +
                "       ,tbl_mock_606.mock_583\n" +
                "       ,tbl_mock_606.mock_584\n" +
                "       ,tbl_mock_606.mock_585\n" +
                "       ,tbl_mock_606.mock_586\n" +
                "       ,tbl_mock_606.mock_587\n" +
                "       ,tbl_mock_606.mock_588\n" +
                "       ,tbl_mock_606.mock_589\n" +
                "       ,tbl_mock_606.mock_591\n" +
                "       ,CASE WHEN (tbl_mock_606.mock_598 > tbl_mock_606.mock_572) THEN (tbl_mock_606.mock_598 * tbl_mock_606.mock_599)  ELSE (CAST(0 AS BIGINT)) END AS mock_602\n" +
                "       ,CASE tbl_mock_606.mock_599 WHEN (CAST(1 AS BIGINT)) THEN tbl_mock_606.mock_590 ELSE NULL END                                                 AS mock_603\n" +
                "       ,(((((((((((tbl_mock_606.mock_578 AND tbl_mock_606.mock_579) AND tbl_mock_606.mock_580) AND tbl_mock_606.mock_581) AND tbl_mock_606.mock_577) AND tbl_mock_606.mock_582) AND tbl_mock_606.mock_583) AND tbl_mock_606.mock_584) AND tbl_mock_606.mock_585) AND tbl_mock_606.mock_586) AND tbl_mock_606.mock_587) AND tbl_mock_606.mock_588) AND tbl_mock_606.mock_589 AS mock_604\n" +
                "FROM tbl_mock_600 AS tbl_mock_606) , tbl_mock_609 (mock_607, mock_608, mock_591) AS (\n" +
                "SELECT  CASE WHEN tbl_mock_610.mock_604 THEN tbl_mock_610.mock_603  ELSE NULL END AS mock_607\n" +
                "       ,tbl_mock_610.mock_604                                                     AS mock_608\n" +
                "       ,tbl_mock_610.mock_591\n" +
                "FROM tbl_mock_605 AS tbl_mock_610) , tbl_mock_611 (mock_591, mock_607) AS (\n" +
                "SELECT  tbl_mock_612.mock_591\n" +
                "       ,tbl_mock_612.mock_607\n" +
                "FROM tbl_mock_609 AS tbl_mock_612\n" +
                "WHERE (tbl_mock_612.mock_591 IS NOT NULL)\n" +
                "AND tbl_mock_612.mock_608) , tbl_mock_614 (mock_591, mock_613) AS (\n" +
                "SELECT  tbl_mock_612.mock_591\n" +
                "       ,COUNT(tbl_mock_612.mock_607) AS mock_613\n" +
                "FROM tbl_mock_611 AS tbl_mock_612\n" +
                "GROUP BY  tbl_mock_612.mock_591)\n" +
                "         ,tbl_mock_617 (mock_615,mock_616) AS (\n" +
                "SELECT  tbl_mock_619.mock_613 AS mock_615\n" +
                "       ,tbl_mock_618.mock_193 AS mock_616\n" +
                "FROM db_mock_000.tbl_mock_192 AS tbl_mock_618\n" +
                "LEFT OUTER JOIN tbl_mock_614 AS tbl_mock_619\n" +
                "ON tbl_mock_618.mock_193 = tbl_mock_619.mock_591) , tbl_mock_621 (mock_616, mock_620) AS (\n" +
                "SELECT  tbl_mock_622.mock_616\n" +
                "       ,coalesce(tbl_mock_622.mock_615,0) AS mock_620\n" +
                "FROM tbl_mock_617 AS tbl_mock_622) , tbl_mock_639 (mock_623, mock_624, mock_625, mock_626, mock_627, mock_628, mock_629, mock_630, mock_631, mock_632, mock_633, mock_634, mock_635, mock_636, mock_637, mock_638) AS (\n" +
                "SELECT  tbl_mock_640.mock_602 AS mock_623\n" +
                "       ,tbl_mock_640.mock_576 AS mock_624\n" +
                "       ,tbl_mock_640.mock_577 AS mock_625\n" +
                "       ,tbl_mock_640.mock_578 AS mock_626\n" +
                "       ,tbl_mock_640.mock_579 AS mock_627\n" +
                "       ,tbl_mock_640.mock_580 AS mock_628\n" +
                "       ,tbl_mock_640.mock_581 AS mock_629\n" +
                "       ,tbl_mock_640.mock_582 AS mock_630\n" +
                "       ,tbl_mock_640.mock_583 AS mock_631\n" +
                "       ,tbl_mock_640.mock_584 AS mock_632\n" +
                "       ,tbl_mock_640.mock_585 AS mock_633\n" +
                "       ,tbl_mock_640.mock_586 AS mock_634\n" +
                "       ,tbl_mock_640.mock_587 AS mock_635\n" +
                "       ,tbl_mock_640.mock_588 AS mock_636\n" +
                "       ,tbl_mock_640.mock_589 AS mock_637\n" +
                "       ,tbl_mock_641.mock_620 AS mock_638\n" +
                "FROM tbl_mock_605 AS tbl_mock_640\n" +
                "LEFT OUTER JOIN tbl_mock_621 AS tbl_mock_641\n" +
                "ON tbl_mock_640.mock_591 = tbl_mock_641.mock_616) , tbl_mock_656 (mock_624, mock_642, mock_643, mock_644, mock_645, mock_646, mock_647, mock_648, mock_649, mock_650, mock_651, mock_652, mock_653, mock_654, mock_655) AS (\n" +
                "SELECT  tbl_mock_657.mock_624\n" +
                "       ,tbl_mock_657.mock_623 / tbl_mock_657.mock_638 AS mock_642\n" +
                "       ,tbl_mock_657.mock_626                         AS mock_643\n" +
                "       ,tbl_mock_657.mock_627                         AS mock_644\n" +
                "       ,tbl_mock_657.mock_628                         AS mock_645\n" +
                "       ,tbl_mock_657.mock_629                         AS mock_646\n" +
                "       ,tbl_mock_657.mock_625                         AS mock_647\n" +
                "       ,tbl_mock_657.mock_630                         AS mock_648\n" +
                "       ,tbl_mock_657.mock_631                         AS mock_649\n" +
                "       ,tbl_mock_657.mock_632                         AS mock_650\n" +
                "       ,tbl_mock_657.mock_633                         AS mock_651\n" +
                "       ,tbl_mock_657.mock_634                         AS mock_652\n" +
                "       ,tbl_mock_657.mock_635                         AS mock_653\n" +
                "       ,tbl_mock_657.mock_636                         AS mock_654\n" +
                "       ,tbl_mock_657.mock_637                         AS mock_655\n" +
                "FROM tbl_mock_639 AS tbl_mock_657) , tbl_mock_659 (mock_624, mock_642, mock_658) AS (\n" +
                "SELECT  tbl_mock_660.mock_624\n" +
                "       ,tbl_mock_660.mock_642\n" +
                "       ,(((((((((((tbl_mock_660.mock_643 AND tbl_mock_660.mock_644) AND tbl_mock_660.mock_645) AND tbl_mock_660.mock_646) AND tbl_mock_660.mock_647) AND tbl_mock_660.mock_648) AND tbl_mock_660.mock_649) AND tbl_mock_660.mock_650) AND tbl_mock_660.mock_651) AND tbl_mock_660.mock_652) AND tbl_mock_660.mock_653) AND tbl_mock_660.mock_654) AND tbl_mock_660.mock_655 AS mock_658\n" +
                "FROM tbl_mock_656 AS tbl_mock_660) , tbl_mock_661 (mock_624, mock_642) AS (\n" +
                "SELECT  tbl_mock_662.mock_624\n" +
                "       ,tbl_mock_662.mock_642\n" +
                "FROM tbl_mock_659 AS tbl_mock_662\n" +
                "WHERE tbl_mock_662.mock_658) , tbl_mock_664 (mock_663) AS (\n" +
                "SELECT  SUM(tbl_mock_665.mock_642) AS mock_663\n" +
                "FROM tbl_mock_661 AS tbl_mock_665\n" +
                "GROUP BY  tbl_mock_665.mock_624)\n" +
                "         ,tbl_mock_668 (mock_663,mock_666,mock_667) AS (\n" +
                "SELECT  tbl_mock_669.mock_663\n" +
                "       ,CASE WHEN (tbl_mock_669.mock_663 < (CAST(0 AS BIGINT))) THEN tbl_mock_669.mock_663  ELSE (CAST(0 AS BIGINT)) END AS mock_666\n" +
                "       ,CASE WHEN (tbl_mock_669.mock_663 > (CAST(0 AS BIGINT))) THEN tbl_mock_669.mock_663  ELSE (CAST(0 AS BIGINT)) END AS mock_667\n" +
                "FROM tbl_mock_664 AS tbl_mock_669) , tbl_mock_670 (mock_663, mock_666, mock_667) AS (\n" +
                "SELECT  tbl_mock_671.mock_663\n" +
                "       ,tbl_mock_671.mock_666\n" +
                "       ,tbl_mock_671.mock_667\n" +
                "FROM tbl_mock_668 AS tbl_mock_671 ORDER BY tbl_mock_671.mock_663 DESC ) , tbl_mock_673 (mock_672) AS (\n" +
                "SELECT  MIN(tbl_mock_674.mock_666) AS mock_672\n" +
                "FROM tbl_mock_670 AS tbl_mock_674) , tbl_mock_676 (mock_675) AS (\n" +
                "SELECT  MAX(tbl_mock_677.mock_667) AS mock_675\n" +
                "FROM tbl_mock_670 AS tbl_mock_677) , tbl_mock_678 (mock_672) AS (\n" +
                "SELECT  tbl_mock_679.mock_672\n" +
                "FROM tbl_mock_673 AS tbl_mock_679\n" +
                "LIMIT 1) , tbl_mock_680 (mock_675) AS (\n" +
                "SELECT  tbl_mock_681.mock_675\n" +
                "FROM tbl_mock_676 AS tbl_mock_681\n" +
                "LIMIT 1) , tbl_mock_684 (mock_682, mock_683) AS (\n" +
                "SELECT  tbl_mock_685.mock_672 AS mock_682\n" +
                "       ,tbl_mock_686.mock_675 AS mock_683\n" +
                "FROM tbl_mock_678 AS tbl_mock_685\n" +
                "CROSS JOIN tbl_mock_680 AS tbl_mock_686 ) , tbl_mock_687 (mock_682, mock_683) AS (\n" +
                "SELECT  tbl_mock_688.mock_682\n" +
                "       ,tbl_mock_688.mock_683\n" +
                "FROM tbl_mock_684 AS tbl_mock_688\n" +
                "LIMIT 50) , tbl_mock_689 (mock_682, mock_683) AS (\n" +
                "SELECT  tbl_mock_690.mock_682\n" +
                "       ,tbl_mock_690.mock_683\n" +
                "FROM tbl_mock_687 AS tbl_mock_690\n" +
                "LIMIT 50) , tbl_mock_691 (mock_194, mock_195, mock_196) AS ((\n" +
                "SELECT  tbl_mock_692.mock_682 AS mock_194\n" +
                "       ,tbl_mock_692.mock_683 AS mock_195\n" +
                "       ,CAST(1 AS BIGINT)     AS mock_196\n" +
                "FROM tbl_mock_689 AS tbl_mock_692)\n" +
                "UNION ALL(\n" +
                "SELECT  NULL              AS mock_194\n" +
                "       ,NULL              AS mock_195\n" +
                "       ,CAST(2 AS BIGINT) AS mock_196\n" +
                "FROM tbl_mock_198 AS tbl_mock_693))\n" +
                "SELECT  tbl_mock_694.mock_194\n" +
                "       ,tbl_mock_694.mock_195\n" +
                "       ,tbl_mock_694.mock_196\n" +
                "FROM tbl_mock_691 AS tbl_mock_694;";
        connectContext.getSessionVariable().setCboCTERuseRatio(0);
        connectContext.getSessionVariable().setCboCTEMaxLimit(20);
        String plan = getFragmentPlan(sql);

        int a = 0;
    }


}
