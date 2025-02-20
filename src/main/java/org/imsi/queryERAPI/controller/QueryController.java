package org.imsi.queryERAPI.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.imsi.queryERAPI.util.PagedResult;
import org.imsi.queryERAPI.util.ResultSetToJsonMapper;
import org.imsi.queryEREngine.imsi.er.QueryEngine;
import org.imsi.queryEREngine.imsi.er.BigVizUtilities.BigVizOutput;
import org.imsi.queryEREngine.imsi.er.Utilities.DumpDirectories;
import org.imsi.queryEREngine.imsi.er.Utilities.SerializationUtilities;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;

import javax.sql.rowset.CachedRowSet;
import javax.sql.rowset.RowSetFactory;
import javax.sql.rowset.RowSetProvider;

import static org.springframework.http.ResponseEntity.ok;

@RestController()
@RequestMapping("/api")
@CrossOrigin
public class QueryController {

	ResultSet rs;
	CachedRowSet rowset;
	List<ObjectNode> results = null;
	DumpDirectories dumpDirectories = new DumpDirectories();
	String query = "";
	@PostMapping("/query")
	public ResponseEntity<String> query(@RequestParam(value = "q", required = true) String q,
			@RequestParam(value = "page", required = false) Integer page, 
			@RequestParam(value = "offset", required = false) Integer offset) throws JsonProcessingException, SQLException  {

		return queryResult(q, page, offset);

	}
	
	@PostMapping("/query-rv")
	public ResponseEntity<String> query(@RequestParam(value = "q", required = true) String q) throws JsonProcessingException, SQLException  {

		return liResult(q);
		
	}
	
	@PostMapping("/columns")
	public ResponseEntity<String> columns(@RequestParam(value = "d", required = true) String dataset) throws JsonProcessingException, SQLException  {
		String q = "SELECT * FROM " + dataset + " LIMIT 3";
		QueryEngine qe = new QueryEngine();
		ObjectMapper mapper = new ObjectMapper();
		rs = qe.runQuery(q);
		ResultSetMetaData rsmd = rs.getMetaData();
		List<String> columns = new ArrayList<>();
		for(int i = 0; i < rsmd.getColumnCount(); i++) {
			columns.add(rsmd.getColumnName(i+1));
		}
		return ok(mapper.writeValueAsString(columns));

	}
	public ResponseEntity<String> liResult(String q) throws SQLException, JsonProcessingException {

		ObjectMapper mapper = new ObjectMapper();
		QueryEngine qe = new QueryEngine();
		if(!this.query.contentEquals(q)) {
			rs = qe.runQuery(q);		
			if(rs != null) {
				BigVizOutput bigVizOutput = (BigVizOutput) SerializationUtilities.loadSerializedObject(dumpDirectories.getLiFilePath());
				return ok(mapper.writeValueAsString(bigVizOutput));
			}
			
		}

		
		return null;
	}
	
	public ResponseEntity<String> queryResult(String q, Integer page, Integer offset) throws SQLException, JsonProcessingException {
		page +=1;

		ObjectMapper mapper = new ObjectMapper();
		QueryEngine qe = new QueryEngine();

		if(!this.query.contentEquals(q)) {
			rs = qe.runQuery(q);		
			RowSetFactory factory = RowSetProvider.newFactory();
			rowset = factory.createCachedRowSet();			 
			rowset.populate(rs);
			this.query = q;
		}

		int end = rowset.size();
		int pages = (int) Math.floor(end / offset) + 1;
		
		int resultOffset = offset * page;
		int startOffset = resultOffset - offset;
		if(page == pages) {
			startOffset = offset * (page - 1);
			resultOffset = end;
			
		}
		if(resultOffset < offset || offset == -1) {
			startOffset = 1;
			resultOffset = end;
		}
		if(startOffset == 0) startOffset = 1;
		results = ResultSetToJsonMapper.mapCRS(rowset, startOffset, resultOffset);

		return ok(mapper.writeValueAsString(new PagedResult(pages, results, end)));
	}
	
	

}
