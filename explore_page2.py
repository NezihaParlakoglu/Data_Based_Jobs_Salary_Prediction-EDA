import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt

def show_explore_page():
    st.title("Explore Data Scientist Salaries")

    st.write(
        """
    ### DS Salaries Dataframe
    """
    )
    # st.set_page_config(page_title="EDA",
    #                    page_icon="bar_chart:",
    #                    layout="wide"
    # )

    # df = pd.read_excel(
    #     io='ds_salaries.xlsx',
    #     engine='openpyxl',
    #     sheet_name="ds_salaries (1)",
    #     skiprows=3,
    #     usecols='B:R',
    #     nrows=1000,
    # )
    df = pd.read_csv("ds_salaries2.csv")
    st.dataframe(df)

    st.sidebar.header("Please Filter Here:")

    Job_title=st.sidebar.multiselect(
        "Select job title:",
        options=["Data Engineer", "Data Scientist", "Data Analyst", "Machine Learning Engineer",
        "Analytics Engineer", "Data Architect"],
        default=["Data Engineer", "Data Scientist", "Data Analyst", "Machine Learning Engineer",
        "Analytics Engineer", "Data Architect"]
    )

    Work_year=st.sidebar.multiselect(
        "Select job title:",
        options=["2020","2021","2022"],
        default=["2020","2021","2022"]
    )

    Remote_ratio=st.sidebar.multiselect(
        "Select remote type:",
        options=df["remote_ratio"].unique(),
        default=df["remote_ratio"].unique()
    )
    print(Remote_ratio)
    Company_loc=st.sidebar.multiselect(
        "Select the company location:",
        options=df["company_location"].unique(),
        default=df["company_location"].unique()
    )


    df_selection = df.query(
        "company_location == @Company_loc & job_title == @Job_title & remote_ratio == @Remote_ratio"
    )

#    st.dataframe(df_selection)
    st.title(":bar_chart: Data Scientist Dashboard")
    total_salary = df_selection["salary_in_usd"].sum()
    avarage_salary = round(df_selection["salary_in_usd"].mean(),1)
    total_employee = round(df_selection["salary_in_usd"].count())

    left_column,middle_column,right_column = st.columns(3)
    with left_column:
        st.subheader("Total Salary:")
        st.subheader(f"US $ {total_salary:,}")
    with middle_column:
        st.subheader("Avarage Salary:")
        st.subheader(f"US $ {avarage_salary}")
    with right_column:
        st.subheader("Total Employee:")
        st.subheader(f":star: {total_employee}")

    st.markdown("---")

    job_title_salary = (
        df_selection.groupby('job_title')['salary_in_usd'].mean().round(0).sort_values(ascending=False).head(15)
    )

    fig_job_title_salary = px.bar(
        job_title_salary,
        x='salary_in_usd',
        y=job_title_salary.index,
        orientation="h",
        title='<b>Average Salary for Top 6 Occupations</b>',
        color_discrete_sequence=['red'] * 3 * len(job_title_salary),
        template="plotly_white",
    )

    fig_job_title_salary.update_layout(
        #        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )

    st.plotly_chart(fig_job_title_salary)



    st.markdown("---")

    job_title_count = (
        df_selection['job_title'].value_counts()
    )

    fig_job_title_count = px.bar(
        job_title_count,
        x = job_title_count.index,
        y = job_title_count,

        orientation="v",
        title = '<b>"Top 6 Popular Jobs"</b>',
        color_discrete_sequence = ['green']*3 * len(job_title_count),
        template = "plotly_white",

    )

    fig_job_title_count.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    employment_type_salary = (
        df_selection.groupby('experience_level')['salary_in_usd'].mean().round(0).nlargest(15).sort_values(ascending = False)
    )

    fig_employment_type_salary = px.bar(
        employment_type_salary,
        x=employment_type_salary.index,
        y='salary_in_usd',
        orientation="v",
        title='<b>Average Salaries by Experience</b>',
        color_discrete_sequence=["#0083B8"] * len( employment_type_salary),
        template="plotly_white",
    )

    fig_employment_type_salary.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )

    left_column, right_column = st.columns(2)
    left_column.plotly_chart(fig_job_title_count, use_container_width=True)
    right_column.plotly_chart(fig_employment_type_salary, use_container_width=True)

    st.markdown("---")

    df["work_year"]=df["work_year"].astype(str)
    employment_type_salary_years = (
        df.groupby(['employment_type','work_year'], as_index=False)['salary_in_usd'].mean().round(0)
    )

    fig_employment_type_salary_years = px.bar(
        employment_type_salary_years,
        x="employment_type",
        y='salary_in_usd',
        color='work_year',
        color_discrete_map={
            '2020': '#3b570f',
            '2021': '#689018',
            '2022': '#1adb4a',
            '2023': '#9be99d'
        },
        barmode='group',
        orientation="v",
        title='<b>Avarage Salaries of Employment Types Based on Years</b>',
        color_discrete_sequence= ["#0083B8"] * len(employment_type_salary_years),
        template="plotly_white",
    )

    fig_employment_type_salary_years.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),

    )

    st.plotly_chart(fig_employment_type_salary_years)

    st.markdown("---")

    avarage_salary_per_year = (
        df.groupby(['work_year'], as_index=False)['salary_in_usd'].mean().round(0)
    )

    fig_avarage_salary_per_year = px.line(avarage_salary_per_year, x="work_year", y="salary_in_usd", title='<b>Avarage Salary Per Years<b>')
    fig_avarage_salary_per_year.update_traces(line_color='#9c412b')
    st.plotly_chart(fig_avarage_salary_per_year)

    #########################################
    rows =["Data Engineer", "Data Scientist", "Data Analyst", "Machine Learning Engineer",
        "Analytics Engineer", "Data Architect"]
#     job_title_salary2 = (
# #        df.groupby('job_title')['salary_in_usd'].sum().round(0)
#         df[df["job_title"].isin(rows)]
#     )
#
#     fig_job_title_salary2 = px.pie(
#         job_title_salary2, values='salary_in_usd', names='job_title', title='<b>Avarage Salary Graph</b>',
#     )

    st.markdown("---")

    # fig_job_title_salary2.update_layout(
    #     #        xaxis=dict(tickmode="linear"),
    #     plot_bgcolor="rgba(0,0,0,0)",
    #     yaxis=(dict(showgrid=False)),
    # )

#    st.plotly_chart(fig_job_title_salary2)

    fig_remote_ratio_percent = px.funnel_area(names=['On-Site', 'Half-Remote', 'Full-Remote'],
                         values=df["remote_ratio"].value_counts(), title='<b>Remote Ratio Graph</b>')

#    st.plotly_chart(fig_employment_type_percent)

    # left_column, right_column = st.columns(2)
    # left_column.plotly_chart(fig_job_title_salary2, use_container_width=True)
    # right_column.plotly_chart(fig_remote_ratio_percent, use_container_width=True)
    st.plotly_chart(fig_remote_ratio_percent)

    st.markdown("---")

    df['country_mean_salary'] = df.groupby(
        ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio',
         'company_location', 'company_size'])['salary_in_usd'].transform('mean')
    rows2 = ["US", "CA", "ES", "DE", "GB", "NG", "IN"]
    fig_sunburst = px.sunburst(df[df["company_location"].isin(rows2)&df["job_title"].isin(rows)], path=['company_location', 'job_title'], values='country_mean_salary', color='job_title',
                hover_data=['experience_level'], height=600,title="<b>Sunburst Graph of Avarage Salaries of Countries Based on Job Titles<b>")

    st.plotly_chart(fig_sunburst)

    st.markdown("---")

    salary_by_country = df.groupby('company_location', as_index=False)['salary_in_usd'].mean()

    fig_salary_by_country = px.choropleth(salary_by_country, locations='company_location', locationmode='country names',
                        color='salary_in_usd',
                        color_continuous_scale='jet', projection='natural earth', hover_name='company_location',
                        labels={'salary_in_usd': 'Average Salary in USD'},
                        title='<b>Distribution of average salary by company location<b>')
    # salary_by_country = df.groupby(['company_location', "work_year"]).agg({'salary_in_usd': "mean"}).reset_index()
    #
    # fig_salary_by_country = px.choropleth(salary_by_country, locations='company_location', locationmode='country names',
    #                     color='salary_in_usd',
    #                     color_continuous_scale='jet', projection='natural earth', hover_name='company_location',
    #                     animation_frame="work_year",
    #                     labels={'salary_in_usd': 'Average Salary in USD'},
    #                     title='<b>Distribution of average salary by company location<b>')

    st.plotly_chart(fig_salary_by_country)

    st.markdown("---")

    # new = df.groupby(["company_location", "work_year", "job_title"]).agg({"salary_in_usd": "mean"})
    # fig_strip_salary_job_title =px.strip(df, y="salary_in_usd", color="job_title", hover_name="company_location", facet_col="company_location",
    #          animation_frame="work_year")

    fig_strip_salary_job_title =px.strip(df[df["job_title"].isin(rows)], y="salary_in_usd", color="experience_level", hover_name="job_title", facet_col="job_title",
             animation_frame="work_year",title="<b>Strip Graph of Avarage Salaries based on Job Titles<b>",width=800)

    st.plotly_chart(fig_strip_salary_job_title)