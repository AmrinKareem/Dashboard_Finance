base_sql = f"""Declare @cBook char(6)='ALL'
        DECLARE @cPeriod char(6)= ?
        DECLARE @segment int = 0 --Set @segment=5141 5001
        DECLARE @iRevNo char(3) ='ALL'
        DECLARE @iProjNo int = 0
        DECLARE @iProjYr int = 0
        DECLARE @Manager tinyint =0
        Declare @ExcludePack  char(3) ='ALL'
        Declare @chkShowClime  char(3) ='0'
        Declare @lcVariation  Nvarchar(10)=''

            if right(@segment, 1) = '1'
            Set    @lcVariation = left(@segment, 3) + '2'
            
            IF OBJECT_ID('tempdb..#tempresult') IS NOT NULL DROP TABLE #tempresult;
        select * into #tempresult from
        (SELECT case when  @cBook='ALL' then 'AUH' else  a.cBook end cBook, a.iProjYear, a.iProjNo, a.cSegment, a.cPackage, a.cPeriod, a.cElementCode,(g.cPackage + ' - ' + g.cSubDesc1) TYP, a.cType, a.iWidth, a.rMthBud,
            a.rMthAct, a.rYearBud, a.rYearBTD, a.rYearAct, a.rYearFrc, a.rPTDBud, a.rPTDAct, a.rOrgBud, a.rRevBud,
            a.rForecast, a.rRateBud, a.rRateFrc, a.rVariance, a.rVarianceChange, l.lCargoBarge,
            (round(a.rForecast,0) -round( a.rLastMonthStcFrc,0)) AS rForecastChange, a.rPercCompl, a.cSubDesc2, a.cSubDesc3, b.lCommitted,
            c.iCurrCode, c.cClient, c.cProjDesc, c.cProjMgr, d.cClientDesc, c.cFirstFrcPeriod,
            e.cCurrAbrv, e.rCurrRate, e.cCurrDesc, f.cDesc as cMajorDesc, g.cAnnex, j.cDesc as cMainDesc, case when  @cBook='ALL' then 'Abu Dhabi' else  k.cBookDesc end cBookDesc
        FROM Ac_JcPackageDt a (NoLock)
        inner join  Ac_JcPackageHd b (NoLock) on a.cBook = b.cBook AND a.iProjYear = b.iProjYear AND a.iProjNo = b.iProjNo AND a.cSegment = b.cSegment AND a.cPackage = b.cPackage AND a.cPeriod = b.cPeriod
        inner join   Ac_JcBudControl c (NoLock) on  a.cBook = c.cBook AND a.iProjYear = c.iProjYear AND a.iProjNo = c.iProjNo AND c.cProjType = 'C'
        left outer join  vwac_Clients d on c.cClient = d.cClientCode
        left outer join       Mm_Currency e (Nolock) on c.iCurrCode = e.iCurrCode
        left outer join        Ac_MainCenter f (NoLock) on  a.cSegment = f.cMajor
        inner join       Ac_JcPackageRef g (NoLock) on  a.cPackage = g.cPackage 
        left outer join    Ac_JcBudControlDetail h (NoLock) on  a.cBook = h.cBook AND a.iProjYear = h.iProjYear AND a.iProjNo = h.iProjNo and a.cSegment = h.cSegment AND h.cProjType = 'C'
        left outer join       Ac_ArSegment j (NoLock) on a.cSegment = j.cCoCode
        left outer join        Cs_Book k (NoLock) on  c.cBook = k.cBook
        left outer join               Ac_JcResources l (NoLock) on  a.cElementCode =l.cResource
        left join  AC_JcMgrSegPack m (Nolock)  on a.cSegment = m.cSegment AND a.cPackage = m.cPackage AND SUBSTRING(a.cElementCode, 1, 2) = m.cCode
        WHERE 
         f.cCompany ='N'
        AND 
	   (
	     (@cBook='ALL')
		 OR
	     (@cBook <> 'ALL' and a.cBook =@cBook)
	   )
       AND a.iProjYear =case when @iProjYr=0 then a.iProjYear else @iProjYr end
       AND a.iProjNo = case when @iProjNo=0 then a.iProjNo else  @iProjNo end
       AND a.cPeriod = @cPeriod
       AND a.cSegment =  case when @segment= 0 then a.cSegment   else   @segment  end 
	   AND  
	   (
	   (@Manager =1 AND a.cSegment = m.cSegment AND a.cPackage = m.cPackage AND SUBSTRING(a.cElementCode, 1, 2) = m.cCode AND  m.iCode=@Manager)
	   or 
	    ( @Manager =0))	   
       AND
	   (( (@Manager =1  AND @segment=0 )  OR  ISNULL(@lcVariation,'')<>'')  AND RIGHT(a.cSegment, 1) <> '2'
	or
		(@Manager =0))
	  AND 	  	  
	  (@iRevNo='ALL' AND  a.cPackage NOT IN ('92','93')
	  OR
       ( a.cPackage = @iRevNo)
	  ))  a        where (a.rMthBuD <> 0 or a.rMthAct <> 0 or a.rYearBud <> 0 or a.rYearBTD <> 0 or a.rYearAct <> 0 or a.rYearFrc <> 0 or a.rPTDBud <> 0 or a.rPTDAct <> 0 or a.rOrgBud <> 0  or a.rRevBud <> 0 or a.rForecast <> 0  or a.rVariance <> 0 or a.rForecastChange <> 0 )
	  if ISNULL(@lcVariation,'')<>''
     BEGIN
	  IF OBJECT_ID('tempdb..#tempresultunion') IS NOT NULL DROP TABLE #tempresultunion;
	   select * into #tempresultunion from
      (SELECT case when  @cBook='ALL' then 'AUH' else  a.cBook end cBook, a.iProjYear, a.iProjNo, a.cSegment, a.cPackage, a.cPeriod, a.cElementCode,(g.cPackage + ' - ' + g.cSubDesc1) TYP, a.cType, a.iWidth, a.rMthBud,
		a.rMthAct, a.rYearBud, a.rYearBTD, a.rYearAct, a.rYearFrc, a.rPTDBud, a.rPTDAct, a.rOrgBud, a.rRevBud,
		a.rForecast, a.rRateBud, a.rRateFrc, a.rVariance, a.rVarianceChange, l.lCargoBarge,
		 (round(a.rForecast,0) -round( a.rLastMonthStcFrc,0)) AS rForecastChange, a.rPercCompl, a.cSubDesc2, a.cSubDesc3, b.lCommitted,
		 c.iCurrCode, c.cClient, c.cProjDesc, c.cProjMgr, d.cClientDesc, c.cFirstFrcPeriod,
		e.cCurrAbrv, e.rCurrRate, e.cCurrDesc, f.cDesc as cMajorDesc, g.cAnnex, j.cDesc as cMainDesc, case when  @cBook='ALL' then 'Abu Dhabi' else  k.cBookDesc end cBookDesc
		FROM Ac_JcPackageDt a (NoLock)
		inner join  Ac_JcPackageHd b (NoLock) on a.cBook = b.cBook AND a.iProjYear = b.iProjYear AND a.iProjNo = b.iProjNo AND a.cSegment = b.cSegment AND a.cPackage = b.cPackage AND a.cPeriod = b.cPeriod
		inner join   Ac_JcBudControl c (NoLock) on  a.cBook = c.cBook AND a.iProjYear = c.iProjYear AND a.iProjNo = c.iProjNo AND c.cProjType = 'C'
		left outer join  vwac_Clients d on c.cClient = d.cClientCode left outer join       Mm_Currency e (Nolock) on c.iCurrCode = e.iCurrCode
		left outer join        Ac_MainCenter f (NoLock) on  a.cSegment = f.cMajor inner join       Ac_JcPackageRef g (NoLock) on  a.cPackage = g.cPackage 
		left outer join    Ac_JcBudControlDetail h (NoLock) on  a.cBook = h.cBook  AND a.iProjYear = h.iProjYear AND a.iProjNo = h.iProjNo and a.cSegment = h.cSegment AND h.cProjType = 'C'
		left outer join       Ac_ArSegment j (NoLock) on a.cSegment = j.cCoCode left outer join        Cs_Book k (NoLock) on  c.cBook = k.cBook
		left outer join               Ac_JcResources l (NoLock) on  a.cElementCode =l.cResource
		left join  AC_JcMgrSegPack m (Nolock)  on a.cSegment = m.cSegment AND a.cPackage = m.cPackage AND SUBSTRING(a.cElementCode, 1, 2) = m.cCode 
		WHERE 
			f.cCompany ='N'
			 AND 
			   (
			     (@cBook='ALL')
			 OR
		     (@cBook <> 'ALL' and a.cBook =@cBook)
			   )
       AND a.iProjYear =case when @iProjYr=0 then a.iProjYear else @iProjYr end
       AND a.iProjNo = case when @iProjNo=0 then a.iProjNo else  @iProjNo end
    AND a.cPeriod = @cPeriod
			AND
			(
			( isnull( @lcVariation,'') <> '' and a.cSegment = @lcVariation)
			OR
			(@lcVariation=0
			))   AND RIGHT(a.cSegment, 1) = '2'	AND 	  	  
		(@iRevNo='ALL' AND  a.cPackage NOT IN ('92','93')
		OR
		( a.cPackage = @iRevNo)
		) ) a        where (a.rMthBuD <> 0 or a.rMthAct <> 0 or a.rYearBud <> 0 or a.rYearBTD <> 0 or a.rYearAct <> 0 or a.rYearFrc <> 0 or a.rPTDBud <> 0 or a.rPTDAct <> 0 or a.rOrgBud <> 0  or a.rRevBud <> 0 or a.rForecast <> 0  or a.rVariance <> 0 or a.rForecastChange <> 0 )		 
	end
				if ( @Manager =1 AND @segment=0)   OR  ISNULL(@lcVariation,'')<>''
			Begin
				select cBook,	iProjYear,	iProjNo,	cSegment,	cPackage,	cPeriod,	cElementCode,	TYP,	cType,	iWidth,
				sum(rMthBud) rMthBud,sum(rMthAct)rMthAct,sum(rYearBud) rYearBud,sum(rYearBTD) rYearBTD, sum(rYearAct) rYearAct,sum(rYearFrc) rYearFrc,sum(rPTDBud) rPTDBud, 
				sum(rPTDAct) rPTDAct, sum(rOrgBud) rOrgBud,sum(rRevBud) rRevBud,sum(rForecast) rForecast,sum(rRateBud) rRateBud,sum(rRateFrc) rRateFrc,sum(rVariance) rVariance,sum(rVarianceChange) rVarianceChange,
				sum(rForecastChange) rForecastChange,sum(rPercCompl) rPercCompl,	cSubDesc2,	cSubDesc3,	lCommitted,	iCurrCode,	cClient,	cProjDesc,	cProjMgr,	cClientDesc,	cFirstFrcPeriod	,cCurrAbrv,
				rCurrRate,	cCurrDesc,	cMajorDesc,	cAnnex	,cMainDesc,	cBookDesc 
				from #tempresult where 1=1 and ( (isnull(@ExcludePack,'ALL')<>'ALL' and cPackage not in ( @ExcludePack)) OR  ((isnull(@ExcludePack,'ALL')= 'ALL' and  cPackage=cPackage)))
			and ((@chkShowClime=0) or (@chkShowClime=1) and   RIGHT(cSegment, 1) = '2') Group by cBook,	iProjYear,	iProjNo,	cSegment,	cPackage,	cPeriod,	cElementCode,	TYP,	cType,	iWidth,
			cSubDesc2,	cSubDesc3,	lCommitted,	iCurrCode,	cClient,	cProjDesc,	cProjMgr,	cClientDesc,	cFirstFrcPeriod	,cCurrAbrv,	rCurrRate,	cCurrDesc,	cMajorDesc,	cAnnex	,cMainDesc,	cBookDesc
			UNION ALL
				select cBook,	iProjYear,	iProjNo,	cSegment,	cPackage,	cPeriod,	cElementCode,	TYP,	cType,	iWidth,
				sum(rMthBud) rMthBud,sum(rMthAct)rMthAct,sum(rYearBud) rYearBud,sum(rYearBTD) rYearBTD, sum(rYearAct) rYearAct,sum(rYearFrc) rYearFrc,sum(rPTDBud) rPTDBud, 
				sum(rPTDAct) rPTDAct, sum(rOrgBud) rOrgBud,sum(rRevBud) rRevBud,sum(rForecast) rForecast,sum(rRateBud) rRateBud,sum(rRateFrc) rRateFrc,sum(rVariance) rVariance,sum(rVarianceChange) rVarianceChange,
				sum(rForecastChange) rForecastChange,sum(rPercCompl) rPercCompl,	cSubDesc2,	cSubDesc3,	lCommitted,	iCurrCode,	cClient,	cProjDesc,	cProjMgr,	cClientDesc,	cFirstFrcPeriod	,cCurrAbrv,
				rCurrRate,	cCurrDesc,	cMajorDesc,	cAnnex	,cMainDesc,	cBookDesc
				from #tempresultunion where 1=1 and ( (isnull(@ExcludePack,'ALL')<>'ALL' and cPackage not in ( @ExcludePack)) OR  ((isnull(@ExcludePack,'ALL')= 'ALL' and  cPackage=cPackage))) 
				and  ((@chkShowClime=0) or (@chkShowClime=1) and   RIGHT(cSegment, 1) = '2') Group by cBook,	iProjYear,	iProjNo,	cSegment,	cPackage,	cPeriod,	cElementCode,	TYP,	cType,	iWidth,
			cSubDesc2,	cSubDesc3,	lCommitted,	iCurrCode,	cClient,	cProjDesc,	cProjMgr,	cClientDesc,	cFirstFrcPeriod	,cCurrAbrv,	rCurrRate,	cCurrDesc,	cMajorDesc,	cAnnex	,cMainDesc,	cBookDesc
			End
			else
			select cBook,	iProjYear,	iProjNo,	cSegment,	cPackage,	cPeriod,	cElementCode,	TYP,	cType,	iWidth,
				sum(rMthBud) rMthBud,sum(rMthAct)rMthAct,sum(rYearBud) rYearBud,sum(rYearBTD) rYearBTD, sum(rYearAct) rYearAct,sum(rYearFrc) rYearFrc,sum(rPTDBud) rPTDBud, 
				sum(rPTDAct) rPTDAct, sum(rOrgBud) rOrgBud,sum(rRevBud) rRevBud,sum(rForecast) rForecast,sum(rRateBud) rRateBud,sum(rRateFrc) rRateFrc,sum(rVariance) rVariance,sum(rVarianceChange) rVarianceChange,
				sum(rForecastChange) rForecastChange,sum(rPercCompl) rPercCompl,	cSubDesc2,	cSubDesc3,	lCommitted,	iCurrCode,	cClient,	cProjDesc,	cProjMgr,	cClientDesc,	cFirstFrcPeriod	,cCurrAbrv,
				rCurrRate,	cCurrDesc,	cMajorDesc,	cAnnex	,cMainDesc,	cBookDesc
				from #tempresult where 1=1 and ( (isnull(@ExcludePack,'ALL')<>'ALL' and cPackage not in ( @ExcludePack)) OR  ((isnull(@ExcludePack,'ALL')= 'ALL' and  cPackage=cPackage)))
				and  ((@chkShowClime=0) or (@chkShowClime=1) and   RIGHT(cSegment, 1) = '2') Group by cBook,	iProjYear,	iProjNo,	cSegment,	cPackage,	cPeriod,	cElementCode,	TYP,	cType,	iWidth,
			cSubDesc2,	cSubDesc3,	lCommitted,	iCurrCode,	cClient,	cProjDesc,	cProjMgr,	cClientDesc,	cFirstFrcPeriod	,cCurrAbrv,	rCurrRate,	cCurrDesc,	cMajorDesc,	cAnnex	,cMainDesc,	cBookDesc
			"""

